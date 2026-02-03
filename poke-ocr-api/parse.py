# parse.py
import re

POKEMON_TYPES = {
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying",
    "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
}

# common OCR junk you showed
BAD_TOKENS = {"mate", "maten", "nortmoi", "mermai", "matst", "o4", "matenl", "maten1"}

LABEL_SNIPPETS = ("stats", "ivs", "evs", "nature", "ability", "item", "markings", "lv")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def clean_species_line(s: str) -> str:
    # keep letters, spaces, apostrophe, hyphen
    s = re.sub(r"[^A-Za-z \-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_level_and_pokemon(texts_all: list[str]):
    """
    Finds the Lv line even when it's 'Lv.' or 'Lvo' etc, and supports:
      - 'Lv. 84 Jirachi'
      - 'Lv. 56' on one line and Pokémon name on next line
    """
    for i, line in enumerate(texts_all):
        t = line.strip()

        # Look for Lv / Lv. / Lvo / Lva etc. anywhere in the line
        if not re.search(r"\bLv\b\.?|\bLvo\b|\bLva\b|\bLvl\b", t, re.IGNORECASE):
            continue

        # Try: same line contains the number
        m2 = re.search(r"(?:Lv\b\.?|Lvo|Lva|Lvl)\s*\.?\s*(\d{1,3})\s*(.*)$", t, re.IGNORECASE)
        if not m2:
            continue

        level = int(m2.group(1))
        rest = (m2.group(2) or "").strip()

        # Case A: species is on the same line (Lv. 84 Jirachi)
        species = clean_species_line(rest)
        if species:
            return level, species

        # Case B: species is on the next line (Lv. 56 \n Hitmontop &)
        if i + 1 < len(texts_all):
            nxt = clean_species_line(texts_all[i + 1])
            if nxt:
                return level, nxt

    return None, ""


def clean_wordish(t: str) -> str:
    # for nickname candidate comparison
    return re.sub(r"[^A-Za-z0-9\-']", "", t).strip()


def clean_ascii_prefix(s: str) -> str:
    """
    Removes weird leading glyphs like ꀎWide Lens -> Wide Lens
    Keeps basic ASCII only (safe for items/moves/labels here).
    """
    if not s:
        return ""
    s = s.strip()
    s = "".join(ch for ch in s if ch.isprintable())
    # keep ASCII only
    s = "".join(ch for ch in s if ord(ch) < 128)
    return re.sub(r"\s+", " ", s).strip()


def clean_item(s: str) -> str:
    # remove (ATK)/(DEF) etc, then strip non-ascii
    s = re.sub(r"\(.*?\)", "", s).strip()
    s = re.sub(r"[^A-Za-z0-9 \-']", "", s).strip()
    s = clean_ascii_prefix(s)
    return "" if s.lower() in {"none", "n/a", "no"} else s


def find_six_slash_numbers(s: str):
    m = re.search(
        r"(\d{1,3})\s*/\s*(\d{1,3})\s*/\s*(\d{1,3})\s*/\s*(\d{1,3})\s*/\s*(\d{1,3})\s*/\s*(\d{1,3})",
        s
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
    return len(s) >= 8  # catches "197791141101..." and "6002520..."


def parse_easyocr_results(results, conf_min=0.60):
    """
    results: list of (bbox, text, conf) from easyocr readtext(..., detail=1)
    Returns dict matching your sheet columns + stats.
    """
    items_all = [(norm(t), float(c)) for (_, t, c) in results if t and norm(t)]
    texts_all = [t for (t, _) in items_all]
    blob = "\n".join(texts_all)

    out = {
        "nickname": "",
        "pokemon": "",
        "item": "",
        "ability": "",
        "level": "",
        "nature": "",

        # NEW: stats
        "stat_hp": 0, "stat_atk": 0, "stat_def": 0, "stat_spa": 0, "stat_spd": 0, "stat_spe": 0,

        "ev_hp": 0, "ev_atk": 0, "ev_def": 0, "ev_spa": 0, "ev_spd": 0, "ev_spe": 0,
        "iv_hp": "", "iv_atk": "", "iv_def": "", "iv_spa": "", "iv_spd": "", "iv_spe": "",
        "move1": "", "move2": "", "move3": "", "move4": "",
        "debug": {
            "raw_lines": texts_all,
            "move_candidates": []
        }
    }

    # Lv. + Pokemon
    lvl, species = extract_level_and_pokemon(texts_all)
    if lvl is not None:
        out["level"] = lvl
    if species:
        out["pokemon"] = species

    # ---------- Nickname (ONLY tokens before the Lv line) ----------
    lv_idx = None
    for i, (t, c) in enumerate(items_all):
        if re.search(r"\bLv\b|Lvo|Lva|Lvl", t, re.IGNORECASE):
            lv_idx = i
            break

    candidates = []
    for i, (t, c) in enumerate(items_all):
        if lv_idx is not None and i >= lv_idx:
            break

        if c < 0.25:
            continue

        tl = t.lower()
        w = clean_wordish(t)
        wl = w.lower()

        if not (2 <= len(w) <= 16):
            continue
        if wl in BAD_TOKENS or wl in POKEMON_TYPES:
            continue
        if any(lbl in tl for lbl in LABEL_SNIPPETS):
            continue
        if out["pokemon"] and wl == out["pokemon"].lower():
            continue

        if w.isdigit() or is_mostly_digits(w):
            continue
        if is_slash_numbers(t):
            continue

        candidates.append((w, c))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        out["nickname"] = candidates[0][0]

    # ---------- Ability ----------
    m = re.search(r"Ability:\s*([A-Za-z][A-Za-z \-']+)", blob, re.IGNORECASE)
    if m:
        out["ability"] = re.sub(r"[^A-Za-z \-']", "", m.group(1)).strip()

    # ---------- Item Held ----------
    m = re.search(r"Item Held:\s*(.+)", blob, re.IGNORECASE)
    if m:
        out["item"] = clean_item(m.group(1))

    # ---------- Nature ----------
    m = re.search(r"Nature:\s*([A-Za-z]+)", blob, re.IGNORECASE)
    if m:
        out["nature"] = m.group(1).strip()

    # ---------- Next-line helper ----------
    def next_after(label: str):
        for i, t in enumerate(texts_all):
            if t.lower().startswith(label.lower()):
                if i + 1 < len(texts_all):
                    return texts_all[i + 1]
        return ""

    # ---------- Stats / IV / EV ----------
    stats_candidate = next_after("Stats:")
    iv_candidate = next_after("IVs:")
    ev_candidate = next_after("EVs:")

    stats = find_six_slash_numbers(stats_candidate) or find_six_slash_numbers(blob)
    ivs = find_six_slash_numbers(iv_candidate) or find_six_slash_numbers(blob)
    evs = find_six_slash_numbers(ev_candidate) or find_six_slash_numbers(blob)

    if stats:
        out["stat_hp"], out["stat_atk"], out["stat_def"], out["stat_spa"], out["stat_spd"], out["stat_spe"] = stats

    if evs:
        out["ev_hp"], out["ev_atk"], out["ev_def"], out["ev_spa"], out["ev_spd"], out["ev_spe"] = evs

    if ivs:
        out["iv_hp"], out["iv_atk"], out["iv_def"], out["iv_spa"], out["iv_spd"], out["iv_spe"] = ivs
    else:
        out["iv_hp"] = out["iv_atk"] = out["iv_def"] = out["iv_spa"] = out["iv_spd"] = out["iv_spe"] = ""

    # ---------- Move candidates ----------
    move_cands = []
    for idx, (t, c) in enumerate(items_all):
        if c < conf_min:
            continue
        tl = t.lower()

        if tl in BAD_TOKENS:
            continue
        if tl in POKEMON_TYPES:
            continue
        if any(k in tl for k in LABEL_SNIPPETS):
            continue
        if re.fullmatch(r"[\d\s/]+", t):
            continue

        parts = t.split()
        if parts and parts[0].islower() and parts[0] in POKEMON_TYPES and len(parts) > 1:
            t = " ".join(parts[1:])

        cleaned = re.sub(r"[^A-Za-z \-']", "", t).strip()
        cleaned = clean_ascii_prefix(cleaned)

        if 3 <= len(cleaned) <= 30:
            move_cands.append((cleaned, float(c), int(idx)))

    out["debug"]["move_candidates"] = move_cands
    return out
