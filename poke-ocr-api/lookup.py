# lookup.py
import json
import re
from rapidfuzz import process, fuzz


def clean_name(s: str) -> str:
    if not s:
        return ""
    # Remove control chars / weird icons safely
    s = "".join(ch for ch in s if ch.isprintable())
    s = s.replace("\u000c", "").replace("\r", "").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Strip trailing tags like "(ATK)", "(DEF)", "(Water)" for matching
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    return s


def norm_key(s: str) -> str:
    """
    Strong normalization for dictionary keys:
    - lowercase
    - keep letters/numbers/spaces/'/-
    - collapse spaces
    """
    s = clean_name(s).lower()
    s = re.sub(r"[^a-z0-9 \-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_json_safe(path: str):
    """
    Loads JSON that may contain illegal control characters.
    Removes ASCII control chars (< 32) except \n, \r, \t before parsing.
    """
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")

    # JSON forbids literal control chars; keep newline/carriage/tab, remove the rest.
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in ("\n", "\r", "\t"))

    return json.loads(text)


def best_match(query: str, choices: list[str], score_cutoff: int = 80):
    q = clean_name(query)
    if not q:
        return "", 0

    hit = process.extractOne(q, choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if not hit:
        return "", 0

    match, score, _idx = hit
    return match, int(score)


class CanonicalLookups:
    def __init__(self, skills_path: str, items_path: str, monsters_path: str):
        skills = load_json_safe(skills_path)
        items = load_json_safe(items_path)
        monsters = load_json_safe(monsters_path)

        self.moves = sorted({clean_name(x.get("name", "")) for x in skills if x.get("name")})
        self.items = sorted({clean_name(x.get("name", "")) for x in items if x.get("name")})
        self.pokemon = sorted({clean_name(m.get("name", "")) for m in monsters if m.get("name")})

        # NEW: canonical pokemon name -> dex id
        # Assumes monsters.json has fields like { "id": 1, "name": "Bulbasaur", ... }
        self.pokemon_name_to_id: dict[str, int] = {}
        self.pokemon_name_to_gender_ratio: dict[str, int] = {}
        for m in monsters:
            name = m.get("name")
            mid = m.get("id")
            if name and isinstance(mid, int):
                key = norm_key(name)
                self.pokemon_name_to_id[key] = mid
                ratio = m.get("gender_ratio")
                if isinstance(ratio, int):
                    self.pokemon_name_to_gender_ratio[key] = ratio

        # Collect abilities from monsters
        abil = set()
        for m in monsters:
            for a in (m.get("abilities") or []):
                if a.get("name"):
                    abil.add(clean_name(a["name"]))
        self.abilities = sorted(abil)

    def get_pokemon_id(self, pokemon_name: str):
        """
        Returns dex id if known, else None.
        Works best if pokemon_name is already canonicalized.
        """
        k = norm_key(pokemon_name)
        return self.pokemon_name_to_id.get(k)

    def get_gender_ratio(self, pokemon_name: str):
        """
        Returns the game's stored gender ratio for a species, or None if unknown.
        Common values:
        - 255: genderless
        - 0: male-only
        - 254: female-only
        """
        k = norm_key(pokemon_name)
        return self.pokemon_name_to_gender_ratio.get(k)

    def canonicalize(self, rec: dict) -> dict:
        # Pokemon
        p, _ = best_match(rec.get("pokemon", ""), self.pokemon, score_cutoff=75)
        if p:
            rec["pokemon"] = p

        # attach pokemon_id (after canonicalization)
        pid = self.get_pokemon_id(rec.get("pokemon", ""))
        if pid is not None:
            rec["pokemon_id"] = pid
        else:
            rec["pokemon_id"] = None

        # Item
        it, _ = best_match(rec.get("item", ""), self.items, score_cutoff=80)
        if it:
            rec["item"] = it

        # Ability
        ab, _ = best_match(rec.get("ability", ""), self.abilities, score_cutoff=75)
        if ab:
            rec["ability"] = ab

        # Moves (optional here; final move picking happens in app.py now)
        for k in ("move1", "move2", "move3", "move4"):
            mv, _ = best_match(rec.get(k, ""), self.moves, score_cutoff=80)
            if mv:
                rec[k] = mv

        return rec
