# parse.py
import re
from typing import List, Tuple, Any

from rapidfuzz import process, fuzz

# -------------------------
# Constants
# -------------------------
POKEMON_TYPES = {
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying",
    "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
}

BAD_TOKENS = {"mate", "maten", "nortmoi", "mermai", "matst", "o4", "matenl", "maten1"}
LABEL_SNIPPETS = ("stats", "ivs", "evs", "nature", "ability", "item", "markings", "lv")

NATURES = [
    "Hardy","Lonely","Brave","Adamant","Naughty",
    "Bold","Docile","Relaxed","Impish","Lax",
    "Timid","Hasty","Serious","Jolly","Naive",
    "Modest","Mild","Quiet","Bashful","Rash",
    "Calm","Gentle","Sassy","Careful","Quirky",
]


# -------------------------
# Basic cleaners
# -------------------------
def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def clean_species_line(s: str) -> str:
    s = re.sub(r"[^A-Za-z \-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_wordish(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-']", "", t).strip()


def clean_ascii_prefix(s: str) -> str:
    if not s:
        return ""
    s = "".join(ch for ch in s if ch.isprintable() and ord(ch) < 128)
    return re.sub(r"\s+", " ", s).strip()


def clean_item(s: str) -> str:
    s = re.sub(r"\(.*?\)", "", s).strip()
    s = re.sub(r"[^A-Za-z0-9 \-']", "", s).strip()
    s = clean_ascii_prefix(s)
    return "" if s.lower() in {"none", "n/a", "no"} else s


# -------------------------
# Parsing helpers
# -------------------------
def extract_level_and_pokemon(texts_all: List[str]):
    for i, line in enumerate(texts_all):
        t = line.strip()

        if not re.search(r"\bLv\b\.?|\bLvo\b|\bLva\b|\bLvl\b", t, re.IGNORECASE):
            continue

        m2 = re.search(r"(?:Lv\b\.?|Lvo|Lva|Lvl)\s*\.?\s*(\d{1,3})\s*(.*)$", t, re.IGNORECASE)
        if not m2:
            continue

        level = int(m2.group(1))
        rest = (m2.group(2) or "").strip()

        species = clean_species_line(rest)
        if species:
            return level, species

        if i + 1 < len(texts_all):
            nxt = clean_species_line(texts_all[i + 1])
            if nxt:
                return level, nxt

    return None, ""


def find_six_slash_numbers(s: str):
    m = re.search(
        r"(\d{1,3})\s*/\s*(\d{1,3})\s*/\s*(\d{1,3})\s*/\s*(\d{1,3})\s*/\s*(\d{1,3})\s*/\s*(\d{1,3})",
        s or ""
    )
    if not m:
        return None
    return [int(m.group(i)) for i in range(1, 7)]


def is_slash_numbers(t: str) -> bool:
    return bool(re.fullmatch(
        r"\d{1,3}\s*/\s*\d{1,3}\s*/\s*\d{1,3}\s*/\s*\d{1,3}\s*/\s*\d{1,3}\s*/\s*\d{1,3}",
        t.strip()
    ))


def is_mostly_digits(t: str) -> bool:
    s = re.sub(r"\D", "", t)
    return len(s) >= 8


# -------------------------
# Nature bbox extraction (robust)
# -------------------------
def _center_y(bbox) -> float:
    ys = [p[1] for p in bbox]
    return sum(ys) / 4.0


def _left_x(bbox) -> float:
    return min(p[0] for p in bbox)


def _right_x(bbox) -> float:
    return max(p[0] for p in bbox)


def _clean_alpha(s: str) -> str:
    return re.sub(r"[^A-Za-z ]", " ", s or "").strip()


def extract_nature_from_boxes(results: List[Tuple[Any, str, float]], y_tol: float = 18.0, min_score: int = 88) -> str:
    """
    Uses EasyOCR bboxes to find the Nature label, then reads the text to the right
    on the same row, and fuzzy-matches ONLY against the 25 legal natures.

    Returns "" if not found.
    """
    if not results:
        return ""

    label_bbox = None

    # Find a token that looks like "Nature" (tolerant to small OCR glitches)
    for bbox, text, conf in results:
        if not text:
            continue
        tl = text.strip().lower()
        if re.search(r"\bnat\w{2,6}\b", tl):
            label_bbox = bbox
            break

    if label_bbox is None:
        return ""

    ly = _center_y(label_bbox)
    lx_right = _right_x(label_bbox)

    # Collect tokens on same horizontal line to the RIGHT of the label
    row_tokens = []
    for bbox, text, conf in results:
        if not text:
            continue
        cy = _center_y(bbox)
        if abs(cy - ly) > y_tol:
            continue
        if _left_x(bbox) <= lx_right + 5:
            continue
        row_tokens.append((_left_x(bbox), text))

    if not row_tokens:
        return ""

    row_tokens.sort(key=lambda t: t[0])
    line = " ".join(t for _, t in row_tokens)

    cleaned = _clean_alpha(line)
    if not cleaned:
        return ""

    hit = process.extractOne(cleaned, NATURES, scorer=fuzz.WRatio, score_cutoff=min_score)
    if not hit:
        return ""

    best, score, _ = hit
    return best


def force_valid_nature(s: str, min_score: int = 90) -> str:
    """
    If we got *something* from regex, force it to be one of the 25 natures.
    Returns "" if it doesn't match well enough.
    """
    s = _clean_alpha(s)
    if not s:
        return ""
    hit = process.extractOne(s, NATURES, scorer=fuzz.WRatio, score_cutoff=min_score)
    return hit[0] if hit else ""


# -------------------------
# Main parser
# -------------------------
def parse_easyocr_results(results, conf_min=0.60):
    """
    results: list of (bbox, text, conf) from easyocr readtext(..., detail=1)
    """
    items_all = [(norm(t), float(c)) for (bbox, t, c) in results if t and norm(t)]
    texts_all = [t for (t, _) in items_all]
    blob = "\n".join(texts_all)

    # -------------------------
    # Confidence helpers
    # -------------------------
    def best_conf_for_substring(substr: str) -> float:
        if not substr:
            return 0.0
        s = substr.lower()
        return max((c for t, c in items_all if s in t.lower()), default=0.0)

    def conf_for_exact_line(line: str) -> float:
        if not line:
            return 0.0
        ln = norm(line)
        for t, c in items_all:
            if t == ln:
                return c
        return 0.0

    out = {
        "nickname": "",
        "pokemon": "",
        "item": "",
        "ability": "",
        "level": "",
        "nature": "",

        "stat_hp": 0, "stat_atk": 0, "stat_def": 0, "stat_spa": 0, "stat_spd": 0, "stat_spe": 0,
        "ev_hp": 0, "ev_atk": 0, "ev_def": 0, "ev_spa": 0, "ev_spd": 0, "ev_spe": 0,
        "iv_hp": "", "iv_atk": "", "iv_def": "", "iv_spa": "", "iv_spd": "", "iv_spe": "",

        "move1": "", "move2": "", "move3": "", "move4": "",

        "debug": {
            "raw_lines": texts_all,
            "move_candidates": [],
            "conf": {
                "nickname": 0.0,
                "pokemon": 0.0,
                "level": 0.0,
                "ability": 0.0,
                "item": 0.0,
                "nature": 0.0,
                "stats": 0.0,
                "ivs": 0.0,
                "evs": 0.0,
            }
        }
    }

    # -------------------------
    # Level + Pokémon
    # -------------------------
    lvl, species = extract_level_and_pokemon(texts_all)
    if lvl is not None:
        out["level"] = lvl
        out["debug"]["conf"]["level"] = best_conf_for_substring("lv")
    if species:
        out["pokemon"] = species
        out["debug"]["conf"]["pokemon"] = best_conf_for_substring(species)

    # -------------------------
    # Nickname
    # -------------------------
    lv_idx = next((i for i, (t, _) in enumerate(items_all)
                   if re.search(r"\bLv\b|Lvo|Lva|Lvl", t, re.IGNORECASE)), None)

    candidates = []
    for i, (t, c) in enumerate(items_all):
        if lv_idx is not None and i >= lv_idx:
            break
        if c < 0.80:
            continue

        w = clean_wordish(t)
        wl = w.lower()

        if not (2 <= len(w) <= 16):
            continue
        if wl in BAD_TOKENS or wl in POKEMON_TYPES:
            continue
        if any(lbl in wl for lbl in LABEL_SNIPPETS):
            continue
        if out["pokemon"] and wl == out["pokemon"].lower():
            continue
        if w.isdigit() or is_mostly_digits(w) or is_slash_numbers(t):
            continue

        candidates.append((w, c))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        out["nickname"] = candidates[0][0]
        out["debug"]["conf"]["nickname"] = candidates[0][1]

    # -------------------------
    # Ability / Item / Nature (tolerant)
    # -------------------------
    # Ability
    m = re.search(r"\bAbil\w*\b\s*[:\-]?\s*([A-Za-z][A-Za-z \-']+)", blob, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        out["ability"] = val
        out["debug"]["conf"]["ability"] = best_conf_for_substring(val)

    # Item
    m = re.search(r"\bItem\b\s*\bHeld\b\s*[:\-]?\s*(.+)", blob, re.IGNORECASE)
    if m:
        val = clean_item(m.group(1))
        out["item"] = val
        out["debug"]["conf"]["item"] = best_conf_for_substring(val)

    # Nature (1) bbox-based (best), then (2) blob regex, then whitelist filter
    bbox_nature = extract_nature_from_boxes(results)
    if bbox_nature:
        out["nature"] = bbox_nature
        out["debug"]["conf"]["nature"] = 1.0  # it's derived from multiple tokens; treat as "confident"
    else:
        m = re.search(r"\bNat\w{2,6}\b\s*[:\-]?\s*([A-Za-z]+)", blob, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            fixed = force_valid_nature(raw)
            out["nature"] = fixed
            out["debug"]["conf"]["nature"] = best_conf_for_substring(raw)

    # -------------------------
    # Stats / IVs / EVs
    # -------------------------
    def next_after(label: str):
        for i, t in enumerate(texts_all):
            if t.lower().startswith(label.lower()):
                return texts_all[i + 1] if i + 1 < len(texts_all) else ""
        return ""

    stats_candidate = next_after("Stats:")
    iv_candidate = next_after("IVs:")
    ev_candidate = next_after("EVs:")

    stats = find_six_slash_numbers(stats_candidate) or find_six_slash_numbers(blob)
    ivs = find_six_slash_numbers(iv_candidate) or find_six_slash_numbers(blob)
    evs = find_six_slash_numbers(ev_candidate) or find_six_slash_numbers(blob)

    if stats:
        out["stat_hp"], out["stat_atk"], out["stat_def"], out["stat_spa"], out["stat_spd"], out["stat_spe"] = stats
        out["debug"]["conf"]["stats"] = conf_for_exact_line(stats_candidate)

    if ivs:
        out["iv_hp"], out["iv_atk"], out["iv_def"], out["iv_spa"], out["iv_spd"], out["iv_spe"] = ivs
        out["debug"]["conf"]["ivs"] = conf_for_exact_line(iv_candidate)

    if evs:
        out["ev_hp"], out["ev_atk"], out["ev_def"], out["ev_spa"], out["ev_spd"], out["ev_spe"] = evs
        out["debug"]["conf"]["evs"] = conf_for_exact_line(ev_candidate)

    # -------------------------
    # Move candidates (with confidence)
    # -------------------------
    for idx, (t, c) in enumerate(items_all):
        if c < conf_min:
            continue

        tl = t.lower()
        if tl in BAD_TOKENS or tl in POKEMON_TYPES:
            continue
        if any(k in tl for k in LABEL_SNIPPETS):
            continue
        if re.fullmatch(r"[\d\s/]+", t):
            continue

        cleaned = clean_ascii_prefix(re.sub(r"[^A-Za-z \-']", "", t).strip())
        if 3 <= len(cleaned) <= 30:
            out["debug"]["move_candidates"].append((cleaned, c, idx))

    return out
