# PokéMMO Screenshot Exporter (OCR Tool)

A web-based OCR tool that extracts Pokémon data from PokéMMO PC screenshots and converts it into structured, reusable formats.

This tool eliminates manual data entry and makes it easy to:

- Build PvP teams
- Export Pokémon to Google Sheets
- Generate PokéPaste sets
- Feed Pokémon data into Firestore-backed applications

---

## Features

- Upload or drag & drop PokéMMO screenshots
- OCR-powered stat extraction using EasyOCR
- Automatic detection of Pokémon species, nickname, level, nature, ability, item, EVs, IVs, and moves
- Canonical correction of Pokémon, move, and item names
- Responsive UI (table on desktop, cards on mobile)
- One-click PokéPaste copy
- CSV export compatible with Google Sheets
- Debug mode with raw OCR output
- Firestore-ready JSON output

---

## Repository Structure

```
pokemmo-ocr/
├─ poke-ocr-api/
│  ├─ app.py
│  ├─ parse.py
│  ├─ lookup.py
│  ├─ data/
│  ├─ requirements.txt
│  └─ run.bat
│
├─ poke-ocr-ui/
│  ├─ index.html
│  └─ assets/
│
├─ .gitignore
└─ README.md
```

---

## Running the Backend (API)

### Docker (Recommended)

From inside `poke-ocr-api/`:

```
docker build -t poke-ocr-api .
docker run --rm --gpus all -e EASYOCR_GPU=1 -p 8000:8000 poke-ocr-api
```

The API will be available at:

```
http://localhost:8000
```

---

## API Endpoints

### Health Check

Simple health check to verify the API is running.

```
GET /health
```

Response:
```
{ "ok": true }
```


### POST /parse

Parses one or more PokéMMO screenshots and returns structured data suitable for:

- UI rendering  
- CSV export  
- PokéPaste generation  

Request:
- multipart/form-data
- Field name: `files`
- Accepts multiple image files

Response:
```
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
```

---

### POST /parse_firestore

Returns Pokémon data formatted for Firestore or other external applications.

Intended for:
- Pokédex apps  
- Collection tracking  
- PvP and team builders  

Request:
- multipart/form-data
- Field name: `files`

Response:
```
{
  "id": "",
  "ownerId": "",
  "species": "Kingdra",
  "nickname": "Kindri",
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
    "Dragon Pulse",
    "Weather Ball",
    "Muddy Water"
  ],
  "notes": "",
  "shiny": null,
  "encounters": 0,
  "gender": "unknown",
  "form": "",
  "secretShiny": null,
  "encounterType": "",
  "ot": null,
  "alpha": null,
  "addedAt": "",
  "pvp": null,
  "e4": null,
  "gymReRuns": null,
  "contestRibbons": null,
  "raidReady": null,
  "collectable": null,
  "eggMoves": null,
  "catchDate": ""
}
```
---

## Frontend Usage

Open directly:

```
poke-ocr-ui/index.html
```

### URL Parameters

| Parameter | Description |
|---------|-------------|
| api | API base URL |
| debug=1 | Enables debug + Firestore mode |

---

## Screenshot Guidelines

- Use default PokéMMO theme
- One Pokémon per screenshot
- Stats, EVs, IVs, moves visible
- No overlays or chat windows
- No cropped or resized images  

---

## Feedback

This project was created for the PokéMMO community to save time and reduce manual work.
Feedback, bug reports, and improvement ideas are welcome.

PokéMMO Forums  
https://forums.pokemmo.com/index.php?/profile/558365-mylis/

Discord  
.Mylis

---

## Disclaimer

Pokémon and PokéMMO are trademarks of their respective owners.
This project is community-made and not affiliated with PokéMMO.