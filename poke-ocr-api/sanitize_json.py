import pathlib

def sanitize(src: str, dst: str):
    p = pathlib.Path(src)
    raw = p.read_bytes()

    # Decode with replacement so we can rewrite safely
    text = raw.decode("utf-8", errors="replace")

    # Remove ASCII control chars except \n \r \t
    # (JSON does not allow literal control chars; they must be escaped)
    cleaned = []
    for ch in text:
        o = ord(ch)
        if o < 32 and ch not in ("\n", "\r", "\t"):
            continue
        cleaned.append(ch)

    out = "".join(cleaned)
    pathlib.Path(dst).write_text(out, encoding="utf-8")

if __name__ == "__main__":
    # adjust paths if your folder name differs
    sanitize("data/items.json", "data/items.clean.json")
    sanitize("data/skills.json", "data/skills.clean.json")
    sanitize("data/monsters.json", "data/monsters.clean.json")
    print("Wrote *.clean.json files to data/")
