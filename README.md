# PokéMMO Screenshot Exporter (OCR Tool)

A web-based OCR tool for PokéMMO that extracts Pokémon data from screenshots or videos and exports it in multiple useful formats.

This tool eliminates manual data entry and makes it easy to:

- Build PvP/Raid teams
- Export Pokémon to Google Sheets
- Generate PokéPaste sets
- Feed Pokémon data into Firestore-backed applications

Live Demo:
https://mylis.github.io/pokemmo_ocr/

Invite the Discord bot:
https://discord.com/oauth2/authorize?client_id=1469456127845466335&permissions=2147600384&integration_type=0&scope=bot+applications.commands

Forum post:
https://forums.pokemmo.com/index.php?/topic/196042-website-pok%C3%A9mmo-ocr-tool-quick-guide/#comment-2214251

---

## Features

- Upload **multiple screenshots** or **one video**
- Automatic frame sampling for videos
- OCR tuned for PokéMMO’s default UI
- Duplicate Pokémon detection (video & batch uploads)
- Shiny detection
- One-click PokéPaste copy
- Export formats:
  - PokéPaste
  - CSV (spreadsheet-friendly)
  - JSON (Firestore-ready)
- No Pokémon data is stored server-side

---

## Supported Inputs

### Screenshots
- PNG / JPG / WebP
- One Pokémon per screenshot
- Default PokéMMO theme strongly recommended 

### Video
- MP4 / WebM / MOV
- Pause ~1 second per Pokémon screen
- Pokémon summary screen only (PC view)

#### Input Guidelines
- Use default PokéMMO theme
- One Pokémon per screenshot
- Stats, EVs, IVs, moves visible
- No overlays or chat windows
- No cropped or resized images 
---

## Output Formats

### PokéPaste
Compatible with PokéPaste / Showdown-style imports.

Supports:
- Full mode (EVs / IVs / Nature)
- OTS (Open Team Sheet) mode

### CSV
Formatted to match common Pokémon tracking spreadsheets.

### JSON
Returned in two modes:
- Raw OCR output
- Firestore-friendly structured format (used by integrations)

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
Index.html
```

### URL Parameters

| Parameter | Description |
|---------|-------------|
| api | API base URL |
| debug=1 | Enables debug + Firestore mode |

---

## Discord Bot (OCR via Slash Commands)

A standalone Discord bot that uses the OCR API and allows users to process screenshots **directly from Discord**.

The bot runs in its **own Docker container** and can point to **any compatible API deployment** (yours or someone else’s).

### Features

- Slash commands only (no message scraping)
- Accepts **image attachments** or **one video**
- Supports **Full** and **OTS (Open Team Sheet)** modes
- Returns:
  - PokéPaste (`.txt`)
  - Firestore-ready JSON (`ocr_json` only)
- Interactive UI:
  - Pokémon selector
  - Toggle OTS mode
  - Copy all / copy selected
- No Pokémon data is stored

### Commands

#### `/ocr`
Full interactive OCR run.

- Upload screenshots or one video as attachments
- Returns:
  - PokéPaste
  - JSON
  - Interactive buttons & selector

#### `/ocr_pokepaste`
Returns **only PokéPaste** as a text file.

- Supports Full / OTS mode
- No JSON output

#### `/ocr_json`
Returns **only JSON** in the following Firestore-ready format  
(fields not detected are left empty):

```json
{
  "id": "",
  "ownerId": "",
  "species": "",
  "nickname": "",
  "level": 0,
  "stats": { "hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0 },
  "evs":   { "hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0 },
  "ivs":   { "hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0 },
  "nature": "",
  "item": "",
  "moves": [],
  "notes": "",
  "shiny": null,
  "gender": "unknown",
  "form": "",
  "alpha": null
}
```
---

## Discord Bot Deployment (Docker)

The Discord bot is deployed separately from the API.

```
Required Environment Variables
DISCORD_TOKEN=your_discord_bot_token
API_BASE=https://api.mylis.net
DEFAULT_MODE=full   # or "ots"
```

Run with Docker
```
docker build -t pokemmo-ocr-discord .
docker run -d \
  --name pokemmo-ocr-discord \
  --env-file .env \
  pokemmo-ocr-discord
```

The bot automatically registers slash commands on startup.

---

## Discord Permissions

When inviting the bot:

### OAuth2 Scopes
- bot
- applications.commands
### Bot Permissions
- Send Messages
- Embed Links
- Attach Files
- Read Message History

No admin permissions required.

--- 

## Frame Extractor (Video Helper Tool)

When uploading videos, you can use the built-in Unique Frame Extractor to automatically detect when a new Pokémon appears.

Instead of manually taking screenshots, the extractor:
- Processes the video frame-by-frame
- Lets you select a small region of the screen (typically the Pokémon name area)
- Compares that region between frames
- Saves a frame only when a meaningful change is detected

This means:
- You get one image per Pokémon
- Duplicate frames are skipped automatically
- You don’t need to manually trim your video

### How It Works

1) Upload a video
2) Drag to select the pokemon preview screen
3) Adjust the difference threshold if needed
4) Extract frames
5) Send frames directly to OCR or download them as ZIP

The original concept for this frame extraction approach was inspired by:
BigCurry – Niche PokeMMO Tools 
https://bigcurry.github.io/Niche-PokeMMO-Tools/
https://github.com/BigCurry/Niche-PokeMMO-Tools

---

## Privacy & Data Handling

This tool **does not log or store Pokémon data**.

**Why?**
- Pokémon teams are sensitive competitive information
- Users should remain in full control of their data
- Keeps hosting costs low and avoids data liability

All uploads are processed in-memory and discarded immediately after processing.

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
This project is community-made and not affiliated with PokéMMO or Pokémon.