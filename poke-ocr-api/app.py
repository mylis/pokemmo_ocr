# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import io
import re
import numpy as np
import cv2
from PIL import Image
import easyocr

from parse import parse_easyocr_results, POKEMON_TYPES, LABEL_SNIPPETS, BAD_TOKENS
from lookup import CanonicalLookups

from rapidfuzz import process, fuzz

app = FastAPI(title="Pokemon OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

lookups = CanonicalLookups(
    skills_path="data/skills.json",
    items_path="data/items.json",
    monsters_path="data/monsters.json",
)

# Init OCR reader once (GPU if available)
import os
import easyocr

USE_GPU = os.getenv("EASYOCR_GPU", "0").lower() in ("1", "true", "yes", "on")
reader = easyocr.Reader(["en"], gpu=USE_GPU)


def ascii_only(s: str) -> str:
    if not s:
        return ""
    s = "".join(ch for ch in str(s) if ord(ch) < 128)
    return s.strip()

def preprocess_for_ui(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

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
            if ll in POKEMON_TYPES:
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

    def prefer_dragon_variants(text_norm: str):
        if "dragon" in text_norm and "pulse" in text_norm:
            return "Dragon Pulse"
        if "dragon" in text_norm and "claw" in text_norm:
            return "Dragon Claw"
        return ""

    scored = []
    for text, conf, idx in merged:
        tnorm = norm_move(text)
        if not tnorm:
            continue

        # ✅ exact match before fuzzy
        if tnorm in canon_set:
            scored.append((canon_set[tnorm], 200, conf, idx))
            continue

        forced = prefer_dragon_variants(tnorm)
        if forced and forced.lower() in canon_set:
            scored.append((canon_set[forced.lower()], 190, conf, idx))
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

def to_firestore_json(parsed: dict, owner_id: str = "") -> dict:
    """
    Builds the JSON shape you provided.
    Anything missing stays "empty":
      - strings: ""
      - numbers: 0
      - booleans: null (Python None)
      - arrays: []
    """
    level = parsed.get("level") if parsed.get("level") not in ("", None) else 0
    try:
        level = int(level)
    except Exception:
        level = 0

    moves = [parsed.get("move1"), parsed.get("move2"), parsed.get("move3"), parsed.get("move4")]
    moves = [m for m in moves if m]  # only actual strings; empty stays empty by being omitted

    return {
        "id": "",                 # keep empty; you can fill client-side or later
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

        # If IVs weren't found you currently store "" — we convert "" -> 0 for this JSON
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
        "shiny": None,
        "encounters": 0,
        "gender": "unknown",
        "form": "",
        "secretShiny": None,
        "encounterType": "",
        "ot": None,
        "alpha": None,
        "addedAt": "",

        # flags (all empty)
        "pvp": None,
        "e4": None,
        "gymReRuns": None,
        "contestRibbons": None,
        "raidReady": None,
        "collectable": None,
        "eggMoves": None,
        "catchDate": ""
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/parse")
async def parse(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    rows = []
    for f in files:
        data = await f.read()
        if not data:
            continue

        img = Image.open(io.BytesIO(data)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        prep = preprocess_for_ui(img_bgr)

        results = reader.readtext(prep, detail=1, paragraph=False)

        parsed = parse_easyocr_results(results, conf_min=0.60)
        parsed = lookups.canonicalize(parsed)

        parsed["item"] = ascii_only(parsed.get("item", ""))

        m1, m2, m3, m4 = pick_moves(parsed, lookups.moves)
        parsed["move1"], parsed["move2"], parsed["move3"], parsed["move4"] = m1, m2, m3, m4

        parsed["source_file"] = f.filename
        rows.append(parsed)

    return {"ok": True, "rows": rows}

@app.post("/parse_firestore")
async def parse_firestore(
    files: list[UploadFile] = File(...),
    ownerId: str = Query(default="", description="Optional user UID to include in ownerId"),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    rows = []
    for f in files:
        data = await f.read()
        if not data:
            continue

        img = Image.open(io.BytesIO(data)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        prep = preprocess_for_ui(img_bgr)

        results = reader.readtext(prep, detail=1, paragraph=False)

        parsed = parse_easyocr_results(results, conf_min=0.60)
        parsed = lookups.canonicalize(parsed)
        parsed["item"] = ascii_only(parsed.get("item", ""))

        m1, m2, m3, m4 = pick_moves(parsed, lookups.moves)
        parsed["move1"], parsed["move2"], parsed["move3"], parsed["move4"] = m1, m2, m3, m4

        rows.append(to_firestore_json(parsed, owner_id=ownerId))

    return {"ok": True, "rows": rows}
