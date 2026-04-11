# Pokemon OCR for PokeMMO

Web UI + OCR API + optional Discord bot for extracting Pokemon data from PokeMMO summary screens.

The project can read screenshots or videos and export:

- PokePaste text
- CSV for spreadsheets
- Firestore-friendly JSON

Links:

- Live demo: https://mylis.github.io/pokemmo_ocr/
- Discord bot invite: https://discord.com/oauth2/authorize?client_id=1469456127845466335&permissions=2147600384&integration_type=0&scope=bot+applications.commands
- Forum post: https://forums.pokemmo.com/index.php?/topic/196042-website-pok%C3%A9mmo-ocr-tool-quick-guide/#comment-2214251

## Current Features

### Web App

- Upload screenshots, videos, or a mix of both in one batch
- Process multiple videos sequentially
- Drag and drop files into the page
- Paste screenshots directly from the clipboard
- Detect unique Pokemon frames from videos before OCR
- Built-in Unique Frame Extractor for full-screen recordings
- Copy individual PokePaste blocks or copy all at once
- Toggle OTS mode in the UI
- Export CSV from standard OCR results
- Send OCR results straight into PokeMMO Box through the transfer flow
- Hide or include the PokePaste column in CSV export
- View Firestore JSON and copy each record or the full JSON bundle
- Delete individual OCR results from the current session
- Responsive desktop table / mobile card layout

### OCR Data

The parser is tuned for the default PokeMMO summary screen and attempts to extract:

- Species
- Nickname
- Level
- Nature
- Ability
- Hidden Ability flag
- Item
- Gender
- Shiny flag
- Alpha flag
- EVs
- IVs
- Stats
- Moves

### Discord Bot

- Slash-command workflow
- Up to 10 screenshot attachments or 1 video per command
- Full and OTS output modes
- Interactive Pokemon picker
- Copy all / copy selected PokePaste output
- JSON export as `rows.json`

## Recommended Inputs

For best OCR accuracy:

- Use the default PokeMMO theme
- Use the Pokemon summary screen
- Keep stats, EVs, IVs, moves, and item visible
- Avoid overlays, chat windows, or heavy cropping
- Use one Pokemon per screenshot
- Keep resolution consistent across uploads when possible

Supported web inputs:

- Images: PNG, JPG, JPEG, WebP
- Videos: MP4, WebM, MOV, MKV

## Web App Usage

### Run Locally

The frontend is a static site. The repo includes a helper batch file:

```bat
run.bat
```

That starts:

```bat
py -3.12 -m http.server 5173
```

Then open:

```text
http://localhost:5173
```

You can also open `index.html` directly, but using the local server is the smoother option for development.

### URL Parameters

| Parameter | Description |
| --- | --- |
| `api` | Override the API base URL. Default is `https://api.mylis.net` |
| `box` | Override the Box frontend URL. Default is `https://box.mylis.net` |
| `boxApi` | Override the Box backend URL used for OCR transfers. Default is `https://backend.mylis.net` |
| `debug=1` | Show Firestore/debug UI, including the Firestore button and extra debug panels |

Example:

```text
http://localhost:5173/?api=http://localhost:8000&box=http://localhost:8080&boxApi=http://localhost:8002&debug=1
```

### Web Workflow

1. Upload screenshots, videos, or a mix of both.
2. The app routes images to image endpoints and videos to video endpoints automatically.
3. Standard mode renders PokePaste-ready results and enables CSV export.
4. Standard mode can also hand the current OCR rows to PokeMMO Box for login, registration, and final import there.
5. Debug mode also exposes Firestore JSON output and extra OCR diagnostics.

## Unique Frame Extractor

The web app includes a built-in helper for recorded videos.

It can:

- Load a video in a modal
- Let you drag-select a region of interest
- Compare that region frame-by-frame
- Save only meaningfully different frames
- Send whole frames or ROI crops directly into OCR
- Send extracted frames to Firestore mode when `debug=1` is enabled
- Download extracted whole-frame or ROI ZIP files

This is useful when you recorded your full screen instead of taking separate screenshots.

Frame extraction inspiration:

- https://bigcurry.github.io/Niche-PokeMMO-Tools/
- https://github.com/BigCurry/Niche-PokeMMO-Tools

## OCR API

The backend lives in `poke-ocr-api/` and is built with FastAPI + EasyOCR.

### Run With Docker

From `poke-ocr-api/`:

```bat
docker build -t poke-ocr-api .
```

CPU example:

```bat
docker run --rm -p 8000:8000 -e EASYOCR_GPU=0 poke-ocr-api
```

GPU example:

```bat
docker run --rm --gpus all -p 8000:8000 -e EASYOCR_GPU=1 poke-ocr-api
```

Windows helper:

```bat
poke-ocr-api\run.bat
```

The API listens on:

```text
http://localhost:8000
```

### Important Environment Variables

- `EASYOCR_GPU`: `1` for GPU, `0` for CPU
- `EASYOCR_MODULE_PATH`: EasyOCR model cache path
- `WEB_CONCURRENCY`: Gunicorn worker count
- `OCR_WORKERS`: OCR thread pool size per worker
- `OCR_INFLIGHT`: Max OCR jobs in flight per worker
- `MAX_UPLOAD_BYTES`: Total request cap, default `250000000`
- `MAX_IMAGE_BYTES`: Per-image cap, default `15000000`
- `MAX_VIDEO_BYTES`: Per-video cap, default `200000000`
- `GUNICORN_TIMEOUT`: Worker timeout

### Endpoints

#### `GET /health`

Returns API status plus GPU/runtime information.

#### `POST /parse`

Standard OCR for one or more images.

- Content type: `multipart/form-data`
- Field name: `files`
- Returns: `{ ok, rows }`

Each row includes standard OCR fields such as:

- `pokemon`
- `nickname`
- `level`
- `nature`
- `ability`
- `item`
- `gender`
- `shiny`
- `alpha`
- `ha`
- `move1` to `move4`
- EV / IV / stat fields
- `source_file`

#### `POST /parse_firestore`

Same image input as `/parse`, but returns Firestore-friendly rows.

- Content type: `multipart/form-data`
- Field name: `files`
- Optional query param: `ownerId`
- Returns: `{ ok, rows }`

#### `POST /parse_video`

OCR for a single video after frame sampling and deduplication.

- Content type: `multipart/form-data`
- Field name: `file`
- Optional query params: `target_fps` (default `3.0`), `dist_threshold` (default `6`), `compare_to` (`last_kept` or `all_kept`), `crop` (`x,y,w,h`)
- Returns: `{ ok, frames_total, frames_unique, rows, rows_unique }`

#### `POST /parse_video_firestore`

Video OCR with Firestore-formatted output.

- Content type: `multipart/form-data`
- Field name: `file`
- Optional query params: `ownerId`, `target_fps`, `dist_threshold`, `compare_to`, `crop`
- Returns: `{ ok, frames_total, frames_unique, rows, rows_unique }`

## Discord Bot

The Discord bot lives in `poke-discord-bot/` and talks to any compatible OCR API deployment.

Current commands:

- `/ocr`
- `/ocr_pokepaste`
- `/ocr_json`
- `/help`
- `/about`
- `/ping`

Bot input rules:

- Up to 10 images or 1 video
- No mixed image + video attachments in the same command
- `DEFAULT_MODE` controls whether the default PokePaste mode is Full or OTS

### Deploy With Docker

From `poke-discord-bot/` create an `.env` file with:

```env
DISCORD_TOKEN=your_discord_bot_token
API_BASE=https://api.mylis.net
DEFAULT_MODE=full
```

Then run:

```bat
docker compose up --build -d
```

Or without Compose:

```bat
docker build -t pokemmo-ocr-discord-bot .
docker run -d --name pokemmo-ocr-discord-bot --env-file .env pokemmo-ocr-discord-bot
```

### Discord Permissions

OAuth scopes:

- `bot`
- `applications.commands`

Bot permissions:

- Send Messages
- Embed Links
- Attach Files
- Read Message History

## Project Layout

```text
.
|-- index.html
|-- css/
|-- js/
|-- assets/
|-- poke-ocr-api/
`-- poke-discord-bot/
```

## Privacy

Pokemon data is not intended to be permanently stored by the web app itself.

- The web frontend sends files to the configured OCR API
- Image uploads are processed in memory
- Video uploads are written to a temporary file for frame sampling and then removed
- The browser keeps current results only for the active page session unless you export/copy them yourself

Host the API yourself if you need full control over privacy, access, logging, or retention.

## Feedback

- Forums: https://forums.pokemmo.com/index.php?/profile/558365-mylis/
- Discord: `.mylis`

## Disclaimer

Pokemon and PokeMMO are trademarks of their respective owners.
This project is community-made and is not affiliated with Pokemon or PokeMMO.
