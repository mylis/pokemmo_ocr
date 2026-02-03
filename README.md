# PokéMMO Screenshot Exporter (OCR Tool)

A web-based OCR tool that extracts Pokémon data from **PokéMMO PC screenshots** and converts it into structured, reusable formats.

This tool eliminates manual data entry and makes it easy to:

- Build PvP teams
- Export Pokémon to Google Sheets
- Generate PokéPaste sets
- Feed Pokémon data into Firestore-backed applications

---

## Features

- Upload or drag & drop PokéMMO screenshots
- OCR-powered stat extraction using **EasyOCR**
- Automatic detection of:
  - Pokémon species
  - Nickname
  - Level
  - Nature
  - Ability
  - Held item
  - EVs / IVs
  - Moves
- Canonical correction of Pokémon, move, and item names
- Responsive UI (table on desktop, cards on mobile)
- One-click **PokéPaste copy**
- **CSV export** (Google Sheets compatible)
- Debug mode with raw OCR output
- Firestore-ready JSON output for external apps

---

## Repository Structure

```
pokemmo-ocr/
├─ poke-ocr-api/        # FastAPI backend
│  ├─ app.py
│  ├─ parse.py
│  ├─ lookup.py
│  ├─ data/
│  ├─ requirements.txt
│  └─ run.bat
│
├─ poke-ocr-ui/         # Static frontend
│  ├─ index.html
│  └─ assets/
│
├─ .gitignore
└─ README.md
```

---

## Running the Backend (API)

Build the image

1. From inside poke-ocr-api/:
docker build -t poke-ocr-api .

2. Run the container
docker run --rm --gpus all -e EASYOCR_GPU=1 -p 8000:8000 poke-ocr-api


The API will be available at:
http://localhost:8000


Health check:
http://localhost:8000/health

---

## API Endpoints
GET /health

Simple health check to verify the API is running.

Response

{
  "ok": true
}

POST /parse

Parses one or more PokéMMO screenshots and returns structured data suitable for:

UI rendering

CSV export

PokéPaste generation

Request

multipart/form-data

Field name: files

Accepts multiple image files

Example

POST /parse
Content-Type: multipart/form-data


Response

{
  "ok": true,
  "rows": [
    {
      "pokemon": "Kingdra",
      "nickname": "dagod",
      "level": 100,
      "nature": "Modest",
      "ability": "Swift Swim",
      "item": "Life Orb",
      "ev_hp": 6,
      "ev_atk": 0,
      "ev_def": 0,
      "ev_spa": 252,
      "ev_spd": 0,
      "ev_spe": 252,
      "iv_hp": 31,
      "iv_atk": 18,
      "iv_def": 31,
      "iv_spa": 31,
      "iv_spd": 31,
      "iv_spe": 31,
      "move1": "Protect",
      "move2": "Muddy Water",
      "move3": "Dragon Pulse",
      "move4": "Weather Ball",
      "debug": {
        "raw_lines": [],
        "move_candidates": []
      }
    }
  ]
}


Notes:

Any value not confidently detected is returned empty or 0

Canonical name matching is applied (moves, items, Pokémon)

Final move selection prioritizes exact matches before fuzzy matching

POST /parse_firestore

Returns Pokémon data formatted for Firestore / external apps.

This endpoint is intended for:

Pokédex apps

Collection tracking

PvP / team management tools

Request

Same as /parse

multipart/form-data

Field name: files

Response

{
  "items": [
    {
      "id": "auto-generated-id",
      "species": "Kingdra",
      "nickname": "dagod",
      "level": 100,
      "stats": {
        "hp": 292,
        "atk": 191,
        "def": 226,
        "spa": 317,
        "spd": 226,
        "spe": 269
      },
      "evs": {
        "hp": 6,
        "atk": 0,
        "def": 0,
        "spa": 252,
        "spd": 0,
        "spe": 252
      },
      "ivs": {
        "hp": 31,
        "atk": 18,
        "def": 31,
        "spa": 31,
        "spd": 31,
        "spe": 31
      },
      "nature": "Modest",
      "item": "Life Orb",
      "moves": [
        "Protect",
        "Muddy Water",
        "Dragon Pulse",
        "Weather Ball"
      ],
      "gender": "unknown"
    }
  ]
}


Notes:

Fields not found by OCR remain empty or defaulted

No UI-specific fields are included

Ideal for direct Firestore insertion

---

## Frontend Usage

Open:
```
poke-ocr-ui/index.html
```

### Query Parameters
| Param | Description |
|-----|-------------|
| `api` | API base URL |
| `debug=1` | Enables debug + Firestore mode |

---

## 📸 Screenshot Guidelines

- Default PokéMMO theme
- One Pokémon per screenshot
- Pokémon summary fully visible
- No overlays or chat
- No chat windows or UI overlays
- No cropped or resized images

---

## Author & Feedback

Built by Mylis

This project was created for the PokéMMO community to save time and reduce manual work.

Feedback, bug reports, and improvement ideas are very welcome:

If OCR fails on a screenshot, feel free to share it

If you have feature ideas (filters, exports, integrations), let me know

If you plan to integrate this into another app, I’d love to hear about it

Contact

🧵 PokéMMO Forums:
https://forums.pokemmo.com/index.php?/profile/558365-mylis/

💬 Discord: .Mylis

## Disclaimer

Pokémon and PokéMMO are trademarks of their respective owners.
This project is community-made and not affiliated with PokéMMO.