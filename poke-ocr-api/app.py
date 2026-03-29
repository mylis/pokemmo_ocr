# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import io
import re
import os
import gc
import tempfile
import threading
from typing import Optional, Tuple, List, Iterable

import numpy as np
import cv2
from PIL import Image
import easyocr

from parse import parse_easyocr_results, POKEMON_TYPES, LABEL_SNIPPETS, BAD_TOKENS
from lookup import CanonicalLookups
from rapidfuzz import process, fuzz

import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# -----------------------------
# Limits & concurrency settings
# -----------------------------
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(250 * 1024 * 1024)))   # total request cap (250MB)
MAX_IMAGE_BYTES  = int(os.getenv("MAX_IMAGE_BYTES",  str(15  * 1024 * 1024)))   # per image cap (15MB)
MAX_VIDEO_BYTES  = int(os.getenv("MAX_VIDEO_BYTES",  str(200 * 1024 * 1024)))   # per video cap (200MB)

OCR_WORKERS  = int(os.getenv("OCR_WORKERS", "4"))   # threads per worker process
OCR_INFLIGHT_ENV = os.getenv("OCR_INFLIGHT")        # max OCR calls at once per worker
OCR_SERIALIZE_GPU = os.getenv("OCR_SERIALIZE_GPU", "1").lower() in ("1", "true", "yes", "on")
OCR_CUDA_CLEANUP_EVERY = int(os.getenv("OCR_CUDA_CLEANUP_EVERY", "100"))

ocr_pool = ThreadPoolExecutor(max_workers=OCR_WORKERS)


# -----------------------------
# Request size limit middleware
# -----------------------------
class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self.max_bytes:
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
            except ValueError:
                pass
        return await call_next(request)


app = FastAPI(title="Pokemon OCR API")

app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_UPLOAD_BYTES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # or set to your frontend origin(s)
    allow_credentials=False,          # MUST be False if allow_origins=["*"]
    allow_methods=["*"],              # includes OPTIONS preflight
    allow_headers=["*"],              # allow Content-Type, Authorization, etc.
    expose_headers=["*"],             # optional: lets browser read extra response headers
    max_age=86400,                    # cache preflight for 24h
)

lookups = CanonicalLookups(
    skills_path="data/skills.json",
    items_path="data/items.json",
    monsters_path="data/monsters.json",
)

lookups.moves_lower = {m.lower() for m in lookups.moves}

USE_GPU = os.getenv("EASYOCR_GPU", "0").lower() in ("1", "true", "yes", "on")

try:
    import torch
except Exception:
    torch = None

OCR_INFLIGHT = int(OCR_INFLIGHT_ENV) if OCR_INFLIGHT_ENV else (1 if USE_GPU else 2)
ocr_sem = asyncio.Semaphore(OCR_INFLIGHT)

reader = easyocr.Reader(
    ["en"],
    gpu=USE_GPU,
    model_storage_directory=os.getenv("EASYOCR_MODULE_PATH", "/opt/easyocr"),
)
gpu_reader_lock = threading.Lock() if (USE_GPU and OCR_SERIALIZE_GPU) else None
gpu_cleanup_lock = threading.Lock()
gpu_calls_since_cleanup = 0

# -----------------------------
# Upload helpers
# -----------------------------
async def read_uploadfile_limited(f: UploadFile, max_bytes: int) -> bytes:
    """
    Safely read an UploadFile into memory with hard cap.
    """
    size = 0
    chunks = []

    while True:
        chunk = await f.read(1024 * 1024)  # 1MB
        if not chunk:
            break

        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (>{max_bytes} bytes)")

        chunks.append(chunk)

    return b"".join(chunks)


async def save_uploadfile_limited(f: UploadFile, dst_path: str, max_bytes: int) -> int:
    size = 0
    async with aiofiles.open(dst_path, "wb") as out:
        while True:
            chunk = await f.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(status_code=413, detail=f"Video too large (>{max_bytes} bytes)")
            await out.write(chunk)
    return size


async def parse_one_image_async(img: Image.Image, source_name: str = "") -> dict:
    """
    Run parse_one_image_pil in a thread (concurrent), but cap inflight OCR.
    """
    async with ocr_sem:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(ocr_pool, parse_one_image_pil, img, source_name)


# -----------------------------
# Small utils
# -----------------------------
def ascii_only(s: str) -> str:
    if not s:
        return ""
    s = "".join(ch for ch in str(s) if ord(ch) < 128)
    return s.strip()


def _has_component(mask: np.ndarray, area_min: int, area_max: int) -> bool:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area_min <= area <= area_max:
            return True
    return False


def _safe_cuda_stats() -> dict:
    if not USE_GPU or torch is None or not torch.cuda.is_available():
        return {"enabled": False}
    try:
        return {
            "enabled": True,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:
        return {"enabled": True, "error": "cuda_stats_unavailable"}


def _cleanup_cuda_cache(force: bool = False) -> None:
    global gpu_calls_since_cleanup

    if not USE_GPU or torch is None or not torch.cuda.is_available():
        return

    with gpu_cleanup_lock:
        if not force:
            gpu_calls_since_cleanup += 1
            if gpu_calls_since_cleanup < max(1, OCR_CUDA_CLEANUP_EVERY):
                return
        gpu_calls_since_cleanup = 0

        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _run_easyocr(prep: np.ndarray):
    if USE_GPU and torch is not None:
        with torch.inference_mode():
            if gpu_reader_lock is not None:
                with gpu_reader_lock:
                    return reader.readtext(prep, detail=1, paragraph=False)
            return reader.readtext(prep, detail=1, paragraph=False)

    return reader.readtext(prep, detail=1, paragraph=False)


# -----------------------------
# Fixed-position icon detectors
# -----------------------------
def detect_alpha_icon(img_bgr: np.ndarray) -> bool:
    """
    Red 'alpha' icon in the top-left.
    Returns python bool (NOT numpy.bool_).
    """
    h, w = img_bgr.shape[:2]
    roi = img_bgr[0:int(0.18*h), 0:int(0.22*w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0,   60,  60])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 60,  60])
    upper2 = np.array([180,255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    return bool(_has_component(mask, area_min=40, area_max=5000))


def detect_shiny(img_bgr: np.ndarray) -> bool:
    """
    Yellow star in the VERY top-left.
    Fix: crop starts at y=0 so it still works when alpha is also present.
    """
    h, w = img_bgr.shape[:2]
    roi = img_bgr[0:int(0.16*h), 0:int(0.22*w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([12, 60, 120])
    upper_yellow = np.array([45,255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    return bool(_has_component(mask, area_min=30, area_max=3000))


# -----------------------------
# Generic HA diamond detector (OCR-anchored)
# -----------------------------
def _bbox_center_y(bbox) -> float:
    ys = [p[1] for p in bbox]
    return float(sum(ys) / 4.0)

def _bbox_right_x(bbox) -> float:
    return float(max(p[0] for p in bbox))

def _bbox_height(bbox) -> float:
    ys = [p[1] for p in bbox]
    return float(max(ys) - min(ys))

def detect_hidden_ability_diamond_generic(img_bgr: np.ndarray, ocr_results) -> bool:
    """
    Generic HA diamond detector:
    - Find 'Ability' label using OCR
    - Search a fixed strip to the right of the LABEL (not the value)
      so ability length / bbox weirdness can't hide the diamond
    - Color mask for teal/cyan (low saturation allowed)
    - Shape check using minAreaRect (more tolerant than 4-point approx)
    """
    h, w = img_bgr.shape[:2]
    if not ocr_results:
        return False

    # 1) locate "Ability" label bbox
    ability_label_bbox = None
    for bbox, text, conf in ocr_results:
        if not text:
            continue
        tl = text.strip().lower()
        # tolerant: Ability / Abil1ty / Abillty...
        if (("abil" in tl) and (len(tl) <= 12)) or re.fullmatch(r"abil\w*", tl):
            ability_label_bbox = bbox
            break

    if ability_label_bbox is None:
        return False

    row_y = _bbox_center_y(ability_label_bbox)
    row_tol = max(18.0, _bbox_height(ability_label_bbox) * 1.6)
    label_right = _bbox_right_x(ability_label_bbox)

    # 2) build ROI strip to the right of the LABEL (covers the value + diamond)
    x1 = int(np.clip(label_right + 6, 0, w - 1))
    # wide enough to cover long abilities + diamond area
    x2 = int(np.clip(label_right + 320, 0, w))

    y1 = int(np.clip(row_y - row_tol * 1.15, 0, h - 1))
    y2 = int(np.clip(row_y + row_tol * 1.15, 0, h))

    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return False

    roi = img_bgr[y1:y2, x1:x2]

    # 3) teal/cyan mask (LOW saturation allowed!)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([70, 5, 55], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    roi_area = roi.shape[0] * roi.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 12 or area > roi_area * 0.25:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw <= 0 or rh <= 0:
            continue

        ar = max(rw, rh) / max(1.0, min(rw, rh))
        if ar > 1.8:  # too skinny to be a diamond
            continue

        # How filled the box is (diamond is ~0.5, but give tolerance)
        extent = area / float(rw * rh)
        if extent < 0.18 or extent > 0.92:
            continue

        # size sanity (diamond is small)
        if max(rw, rh) < 6 or max(rw, rh) > 40:
            continue

        return True

    return False


# -----------------------------
# Image preprocessing for OCR
# -----------------------------
def preprocess_color_for_ui(img_bgr: np.ndarray) -> np.ndarray:
    """
    Match the same geometry used for OCR (2x upscaling).
    Keep it in BGR so we can do HSV color detection aligned to OCR bboxes.
    """
    return cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def preprocess_for_ui(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


# -----------------------------
# Move picker
# -----------------------------
def pick_moves(parsed: dict, move_choices: list[str]) -> tuple[str, str, str, str]:
    raw_lines = [str(x) for x in parsed.get("debug", {}).get("raw_lines", [])]
    markings_idx = None
    for i, line in enumerate(raw_lines):
        if line.lower().startswith("markings"):
            markings_idx = i
            break

    cands = parsed.get("debug", {}).get("move_candidates", [])  # (text, conf, idx)

    if markings_idx is not None:
        cands = [c for c in cands if c[2] > markings_idx]
    else:
        cands = list(cands)

    fallback = []
    if markings_idx is not None:
        for j in range(markings_idx + 1, min(markings_idx + 25, len(raw_lines))):
            line = raw_lines[j].strip()
            if not line:
                continue

            ll = line.lower()
            if ll in POKEMON_TYPES and ll not in lookups.moves_lower:
                continue
            if any(k in ll for k in LABEL_SNIPPETS):
                continue
            if ll in BAD_TOKENS:
                continue
            if re.fullmatch(r"[\d\s/]+", line):
                continue

            parts = line.split()
            if parts and parts[0].lower() in POKEMON_TYPES and len(parts) > 1:
                line = " ".join(parts[1:])

            cleaned = re.sub(r"[^A-Za-z \-']", "", line).strip()
            if 3 <= len(cleaned) <= 30:
                fallback.append((cleaned, 0.20, j))

    merged = cands + fallback

    canon_set = {m.lower(): m for m in move_choices}

    def norm_move(s: str) -> str:
        s = re.sub(r"[^a-zA-Z \-']", " ", str(s))
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    scored = []
    for text, conf, idx in merged:
        tnorm = norm_move(text)
        if not tnorm:
            continue

        # exact match before fuzzy
        if tnorm in canon_set:
            scored.append((canon_set[tnorm], 200, conf, idx))
            continue

        hit = process.extractOne(text, move_choices, scorer=fuzz.WRatio, score_cutoff=88)
        if not hit:
            continue
        best, score, _ = hit
        scored.append((best, score, conf, idx))

    scored.sort(key=lambda x: (-x[1], -x[2], x[3]))

    picked = []
    used = set()
    for best, score, conf, idx in scored:
        if best in used:
            continue
        used.add(best)
        picked.append(best)
        if len(picked) == 4:
            break

    while len(picked) < 4:
        picked.append("")

    return picked[0], picked[1], picked[2], picked[3]


# -----------------------------
# Firestore format
# -----------------------------
def to_firestore_json(parsed: dict, owner_id: str = "") -> dict:
    level = parsed.get("level") if parsed.get("level") not in ("", None) else 0
    try:
        level = int(level)
    except Exception:
        level = 0

    moves = [parsed.get("move1"), parsed.get("move2"), parsed.get("move3"), parsed.get("move4")]
    moves = [m for m in moves if m]

    return {
        "id": "",
        "ownerId": owner_id or "",
        "species": parsed.get("pokemon") or "",
        "nickname": parsed.get("nickname") or "",
        "level": level,
        "stats": {
            "hp": int(parsed.get("stat_hp") or 0),
            "atk": int(parsed.get("stat_atk") or 0),
            "def": int(parsed.get("stat_def") or 0),
            "spa": int(parsed.get("stat_spa") or 0),
            "spd": int(parsed.get("stat_spd") or 0),
            "spe": int(parsed.get("stat_spe") or 0),
        },
        "evs": {
            "hp": int(parsed.get("ev_hp") or 0),
            "atk": int(parsed.get("ev_atk") or 0),
            "def": int(parsed.get("ev_def") or 0),
            "spa": int(parsed.get("ev_spa") or 0),
            "spd": int(parsed.get("ev_spd") or 0),
            "spe": int(parsed.get("ev_spe") or 0),
        },
        "ivs": {
            "hp": int(parsed.get("iv_hp") or 0),
            "atk": int(parsed.get("iv_atk") or 0),
            "def": int(parsed.get("iv_def") or 0),
            "spa": int(parsed.get("iv_spa") or 0),
            "spd": int(parsed.get("iv_spd") or 0),
            "spe": int(parsed.get("iv_spe") or 0),
        },
        "nature": parsed.get("nature") or "",
        "item": parsed.get("item") or "",
        "moves": moves,
        "notes": "",
        "shiny": bool(parsed.get("shiny", False)),
        "encounters": 0,
        "gender": "unknown",
        "form": "",
        "secretShiny": None,
        "encounterType": "",
        "ot": None,
        "alpha": bool(parsed.get("alpha", False)),
        "addedAt": "",
        "pvp": None,
        "e4": None,
        "gymReRuns": None,
        "contestRibbons": None,
        "raidReady": None,
        "collectable": None,
        "eggMoves": None,
        "catchDate": "",
    }


# -----------------------------
# Shared image parsing pipeline
# -----------------------------
def parse_one_image_pil(img: Image.Image, source_name: str = "") -> dict:
    img_rgb = img.convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

    parsed = {}

    # Fixed-position icons are OK on original
    parsed["shiny"] = bool(detect_shiny(img_bgr))
    parsed["alpha"] = bool(detect_alpha_icon(img_bgr))

    prep = preprocess_for_ui(img_bgr)
    img_bgr_big = preprocess_color_for_ui(img_bgr)
    results = _run_easyocr(prep)
    _cleanup_cuda_cache()

    parsed["ha"] = bool(detect_hidden_ability_diamond_generic(img_bgr_big, results))

    parsed.update(parse_easyocr_results(results, conf_min=0.60))
    parsed = lookups.canonicalize(parsed)
    parsed["item"] = ascii_only(parsed.get("item", ""))

    m1, m2, m3, m4 = pick_moves(parsed, lookups.moves)
    parsed["move1"], parsed["move2"], parsed["move3"], parsed["move4"] = m1, m2, m3, m4

    if source_name:
        parsed["source_file"] = source_name

    img_rgb.close()
    del img_bgr
    del img_bgr_big
    del prep
    del results

    return parsed

# -----------------------------
# Video -> frames -> unique
# -----------------------------
def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """
    dHash: resize to (hash_size+1, hash_size), compare adjacent columns.
    Returns a 64-bit int when hash_size=8.
    """
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()

    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def iter_sampled_frames(
    video_path: str,
    target_fps: float = 3.0,
    crop: Optional[Tuple[int, int, int, int]] = None,  # x,y,w,h
) -> Iterable[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / max(target_fps, 0.1))))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i % step != 0:
            i += 1
            continue
        i += 1

        if crop is not None:
            x, y, w, h = crop
            frame = frame[y : y + h, x : x + w]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield Image.fromarray(rgb)

    cap.release()


def keep_unique_frames(
    frames: Iterable[Image.Image],
    dist_threshold: int = 6,
    compare_to: str = "last_kept",  # "last_kept" (fast) or "all_kept" (strict)
    hash_size: int = 8,
) -> tuple[List[Image.Image], int]:
    kept: List[Image.Image] = []
    kept_hashes: List[int] = []
    frames_total = 0

    for img in frames:
        frames_total += 1
        h = dhash(img, hash_size=hash_size)

        if not kept:
            kept.append(img)
            kept_hashes.append(h)
            continue

        if compare_to == "all_kept":
            if all(hamming_distance(h, kh) >= dist_threshold for kh in kept_hashes):
                kept.append(img)
                kept_hashes.append(h)
            else:
                img.close()
        else:
            if hamming_distance(h, kept_hashes[-1]) >= dist_threshold:
                kept.append(img)
                kept_hashes.append(h)
            else:
                img.close()

    return kept, frames_total


def parse_crop_param(crop_raw: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not crop_raw:
        return None
    parts = [p.strip() for p in crop_raw.split(",")]
    if len(parts) != 4:
        raise ValueError("crop must be 'x,y,w,h'")
    x, y, w, h = map(int, parts)
    if x < 0 or y < 0:
        raise ValueError("crop x/y must be >= 0")
    if w <= 0 or h <= 0:
        raise ValueError("crop width/height must be > 0")
    return (x, y, w, h)


def parsed_signature(parsed: dict) -> tuple:
    def norm_str(x) -> str:
        return ascii_only(str(x or "")).strip().lower()

    def norm_int(x) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    def norm_bool(x) -> int:
        return 1 if x else 0

    moves = (
        norm_str(parsed.get("move1")),
        norm_str(parsed.get("move2")),
        norm_str(parsed.get("move3")),
        norm_str(parsed.get("move4")),
    )

    sig = (
        norm_str(parsed.get("pokemon")),
        norm_int(parsed.get("level")),
        norm_str(parsed.get("nature")),
        norm_str(parsed.get("ability")),
        norm_str(parsed.get("item")),

        norm_int(parsed.get("ev_hp")),
        norm_int(parsed.get("ev_atk")),
        norm_int(parsed.get("ev_def")),
        norm_int(parsed.get("ev_spa")),
        norm_int(parsed.get("ev_spd")),
        norm_int(parsed.get("ev_spe")),

        norm_int(parsed.get("iv_hp")),
        norm_int(parsed.get("iv_atk")),
        norm_int(parsed.get("iv_def")),
        norm_int(parsed.get("iv_spa")),
        norm_int(parsed.get("iv_spd")),
        norm_int(parsed.get("iv_spe")),

        norm_bool(parsed.get("shiny")),
        norm_bool(parsed.get("alpha")),
        norm_bool(parsed.get("ha")),

        moves,
    )
    return sig


def mon_identity_key(parsed: dict) -> tuple:
    def ns(x: str) -> str:
        return ascii_only(str(x or "")).strip().lower()

    def ni(x) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    return (
        ns(parsed.get("pokemon")),
        ni(parsed.get("level")),
        ns(parsed.get("nature")),
        1 if parsed.get("shiny") else 0,
        1 if parsed.get("alpha") else 0,
        1 if parsed.get("ha") else 0,
    )


def parsed_quality_score(parsed: dict) -> int:
    score = 0

    for k in ("pokemon", "nature", "ability", "item", "move1", "move2", "move3", "move4"):
        if (parsed.get(k) or "").strip():
            score += 5

    num_keys = (
        "level",
        "ev_hp","ev_atk","ev_def","ev_spa","ev_spd","ev_spe",
        "iv_hp","iv_atk","iv_def","iv_spa","iv_spd","iv_spe",
        "stat_hp","stat_atk","stat_def","stat_spa","stat_spd","stat_spe",
    )
    for k in num_keys:
        v = parsed.get(k)
        try:
            if int(v) != 0:
                score += 1
        except Exception:
            pass

    score += 2 if parsed.get("shiny") else 0
    score += 2 if parsed.get("alpha") else 0
    score += 2 if parsed.get("ha") else 0

    return score


def dedupe_keep_best(parsed_list: list[dict]) -> list[dict]:
    seen_sig = set()
    exact_unique = []
    for p in parsed_list:
        sig = parsed_signature(p)
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        exact_unique.append(p)

    best_by_id: dict[tuple, tuple[int, int, dict]] = {}
    for idx, p in enumerate(exact_unique):
        mid = mon_identity_key(p)
        sc = parsed_quality_score(p)

        if mid not in best_by_id:
            best_by_id[mid] = (sc, idx, p)
        else:
            prev_sc, prev_idx, prev_p = best_by_id[mid]
            if sc > prev_sc:
                best_by_id[mid] = (sc, prev_idx, p)

    out = sorted(best_by_id.values(), key=lambda t: t[1])
    return [p for _, _, p in out]


@app.get("/health")
def health():
    return {
        "ok": True,
        "gpu": {
            "use_gpu": USE_GPU,
            "serialize_gpu": OCR_SERIALIZE_GPU,
            "cuda_cleanup_every": OCR_CUDA_CLEANUP_EVERY,
            "inflight_limit": OCR_INFLIGHT,
            "torch_present": torch is not None,
            "cuda": _safe_cuda_stats(),
        },
    }


# -----------------------------
# Existing endpoints (refactored)
# -----------------------------
@app.post("/parse")
async def parse(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    async def one_file(f: UploadFile):
        data = await read_uploadfile_limited(f, MAX_IMAGE_BYTES)
        if not data:
            return None
        with Image.open(io.BytesIO(data)) as im:
            img = im.convert("RGB")
        try:
            return await parse_one_image_async(img, source_name=f.filename or "")
        finally:
            img.close()

    out = await asyncio.gather(*[one_file(f) for f in files])
    rows = [r for r in out if r is not None]
    return {"ok": True, "rows": rows}


@app.post("/parse_firestore")
async def parse_firestore(
    files: list[UploadFile] = File(...),
    ownerId: str = Query(default="", description="Optional user UID to include in ownerId"),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    async def one_file(f: UploadFile):
        data = await read_uploadfile_limited(f, MAX_IMAGE_BYTES)
        if not data:
            return None
        with Image.open(io.BytesIO(data)) as im:
            img = im.convert("RGB")
        try:
            parsed = await parse_one_image_async(img, source_name=f.filename or "")
            return to_firestore_json(parsed, owner_id=ownerId)
        finally:
            img.close()

    tasks = [one_file(f) for f in files]
    out = await asyncio.gather(*tasks)
    rows = [r for r in out if r is not None]

    return {"ok": True, "rows": rows}


@app.post("/parse_video")
async def parse_video(
    file: UploadFile = File(...),
    target_fps: float = Query(default=3.0, description="Frames per second to sample from video"),
    dist_threshold: int = Query(default=6, description="dHash Hamming distance threshold for uniqueness"),
    compare_to: str = Query(default="last_kept", description="'last_kept' or 'all_kept'"),
    crop: str = Query(default="", description="Optional crop 'x,y,w,h' before hashing/OCR (recommended)"),
):
    if not file:
        raise HTTPException(status_code=400, detail="No video uploaded")

    if compare_to not in ("last_kept", "all_kept"):
        raise HTTPException(status_code=400, detail="compare_to must be 'last_kept' or 'all_kept'")

    try:
        crop_tuple = parse_crop_param(crop) if crop else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tmp_path = tf.name

        await save_uploadfile_limited(file, tmp_path, MAX_VIDEO_BYTES)

        unique_frames, frames_total = keep_unique_frames(
            iter_sampled_frames(tmp_path, target_fps=target_fps, crop=crop_tuple),
            dist_threshold=dist_threshold,
            compare_to=compare_to,
        )

        parsed_list = []
        for idx, img in enumerate(unique_frames):
            try:
                parsed = await parse_one_image_async(img, source_name=f"{file.filename or 'video'}#frame{idx}")
                parsed_list.append(parsed)
            finally:
                img.close()

        rows = dedupe_keep_best(parsed_list)

        return {
            "ok": True,
            "frames_total": frames_total,
            "frames_unique": len(unique_frames),
            "rows": rows,
            "rows_unique": len(rows),
        }

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


@app.post("/parse_video_firestore")
async def parse_video_firestore(
    file: UploadFile = File(...),
    ownerId: str = Query(default="", description="Optional user UID to include in ownerId"),
    target_fps: float = Query(default=3.0, description="Frames per second to sample from video"),
    dist_threshold: int = Query(default=6, description="dHash Hamming distance threshold for uniqueness"),
    compare_to: str = Query(default="last_kept", description="'last_kept' or 'all_kept'"),
    crop: str = Query(default="", description="Optional crop 'x,y,w,h' before hashing/OCR (recommended)"),
):
    if not file:
        raise HTTPException(status_code=400, detail="No video uploaded")

    if compare_to not in ("last_kept", "all_kept"):
        raise HTTPException(status_code=400, detail="compare_to must be 'last_kept' or 'all_kept'")

    try:
        crop_tuple = parse_crop_param(crop) if crop else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tmp_path = tf.name

        await save_uploadfile_limited(file, tmp_path, MAX_VIDEO_BYTES)

        unique_frames, frames_total = keep_unique_frames(
            iter_sampled_frames(tmp_path, target_fps=target_fps, crop=crop_tuple),
            dist_threshold=dist_threshold,
            compare_to=compare_to,
        )

        parsed_list = []
        for idx, img in enumerate(unique_frames):
            try:
                parsed = await parse_one_image_async(img, source_name=f"{file.filename or 'video'}#frame{idx}")
                parsed_list.append(parsed)
            finally:
                img.close()

        parsed_unique = dedupe_keep_best(parsed_list)
        rows = [to_firestore_json(p, owner_id=ownerId) for p in parsed_unique]

        return {
            "ok": True,
            "frames_total": frames_total,
            "frames_unique": len(unique_frames),
            "rows": rows,
            "rows_unique": len(rows),
        }

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
