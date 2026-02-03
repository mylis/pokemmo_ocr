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

- `GET /health`
- `POST /parse`
- `POST /parse_firestore`

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
- No overlays or chat

---

## Author

**Mylis**
- Feel free to contact me for feedback
- PokéMMO Forums: https://forums.pokemmo.com/index.php?/profile/558365-mylis/
- Discord: `.Mylis`
