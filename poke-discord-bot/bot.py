# bot.py
import os
import io
import json
from typing import List, Tuple, Optional

import aiohttp
import discord
from discord import app_commands

# -----------------------------
# Config (ENV)
# -----------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
API_BASE = os.getenv("API_BASE", "").strip().rstrip("/")  # e.g. https://api.mylis.net
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "full").strip().lower()  # "full" | "ots"

if not DISCORD_TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN env var")
if not API_BASE:
    raise RuntimeError("Missing API_BASE env var (e.g. https://api.mylis.net)")

# Limits / safety
MAX_ATTACHMENTS = 10
HTTP_TIMEOUT_SECONDS = 120

IMAGE_MIME_PREFIX = "image/"
VIDEO_MIME_PREFIX = "video/"

# -----------------------------
# Helpers
# -----------------------------
def is_image(att: discord.Attachment) -> bool:
    ctype = (att.content_type or "").lower()
    return ctype.startswith(IMAGE_MIME_PREFIX) or att.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

def is_video(att: discord.Attachment) -> bool:
    ctype = (att.content_type or "").lower()
    return ctype.startswith(VIDEO_MIME_PREFIX) or att.filename.lower().endswith((".mp4", ".mov", ".webm", ".mkv"))

def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def collect_attachments(*atts: Optional[discord.Attachment]) -> List[discord.Attachment]:
    return [a for a in atts if a is not None]

def to_stat_block(hp=0, atk=0, defe=0, spa=0, spd=0, spe=0) -> dict:
    return {"hp": hp, "atk": atk, "def": defe, "spa": spa, "spd": spd, "spe": spe}

def normalize_pokemon_row(row: dict) -> dict:
    # API differences:
    # - /parse uses pokemon, /parse_firestore may use species
    species = (row.get("species") or row.get("pokemon") or "").strip()
    nickname = (row.get("nickname") or "").strip()
    nature = (row.get("nature") or "").strip()
    item = (row.get("item") or "").strip()

    level = safe_int(row.get("level"), 0)

    # EVs / IVs (your API uses ev_hp, ev_atk... and iv_hp, iv_atk...)
    evs = to_stat_block(
        hp=safe_int(row.get("ev_hp"), 0),
        atk=safe_int(row.get("ev_atk"), 0),
        defe=safe_int(row.get("ev_def"), 0),
        spa=safe_int(row.get("ev_spa"), 0),
        spd=safe_int(row.get("ev_spd"), 0),
        spe=safe_int(row.get("ev_spe"), 0),
    )

    ivs = to_stat_block(
        hp=safe_int(row.get("iv_hp"), 0),
        atk=safe_int(row.get("iv_atk"), 0),
        defe=safe_int(row.get("iv_def"), 0),
        spa=safe_int(row.get("iv_spa"), 0),
        spd=safe_int(row.get("iv_spd"), 0),
        spe=safe_int(row.get("iv_spe"), 0),
    )

    # Moves array
    moves = []
    for k in ("move1", "move2", "move3", "move4"):
        mv = (row.get(k) or "").strip()
        if mv:
            moves.append(mv)

    # id: use what you have, otherwise empty
    # (your /parse sometimes has pokemon_id; Firestore might have id)
    pid_raw = row.get("id") or row.get("pokemon_id") or ""
    pid = str(pid_raw).strip() if pid_raw is not None else ""


    shiny = bool(row.get("shiny", False))

    # Everything else: leave empty defaults
    normalized = {
        "id": pid,
        "ownerId": "",
        "species": species,
        "nickname": nickname,
        "level": level,

        # stats not provided by your API today -> empty (0s)
        "stats": to_stat_block(),

        "evs": evs,
        "ivs": ivs,

        "nature": nature,
        "item": item,
        "moves": moves,
        "notes": "",

        "shiny": shiny,
        "encounters": 0,
        "gender": "unknown",
        "form": "",
        "secretShiny": False,
        "encounterType": "",
        "ot": False,
        "alpha": False,
        "addedAt": "",

        "pvp": False,
        "e4": False,
        "gymReRuns": False,
        "contestRibbons": False,
        "raidReady": False,
        "collectable": False,
        "eggMoves": False,
        "catchDate": ""
    }

    return normalized


def build_pokepaste(row: dict, ots: bool) -> str:
    nickname = (row.get("nickname") or "").strip()
    species = str(row.get("species") or row.get("pokemon") or "").strip()
    nickname = str(row.get("nickname") or "").strip()
    nature = str(row.get("nature") or "").strip()
    item = str(row.get("item") or "").strip()

    level = safe_int(row.get("level"), 0)
    shiny = bool(row.get("shiny", False))

    def get_num(key: str) -> int:
        return safe_int(row.get(key), 0)

    def get_iv(key: str) -> Optional[int]:
        v = row.get(key)
        if v is None or v == "":
            return None
        return safe_int(v, 0)

    if nickname:
        head = f"{nickname} ({species})"
    else:
        head = species

    if item:
        head = f"{head} @ {item}"

    lines: List[str] = [head.strip()]
    if ability:
        lines.append(f"Ability: {ability}")
    if level:
        lines.append(f"Level: {level}")
    if shiny:
        lines.append("Shiny: Yes")

    if not ots:
        ev_parts = []
        if get_num("ev_hp"):  ev_parts.append(f"{get_num('ev_hp')} HP")
        if get_num("ev_atk"): ev_parts.append(f"{get_num('ev_atk')} Atk")
        if get_num("ev_def"): ev_parts.append(f"{get_num('ev_def')} Def")
        if get_num("ev_spa"): ev_parts.append(f"{get_num('ev_spa')} SpA")
        if get_num("ev_spd"): ev_parts.append(f"{get_num('ev_spd')} SpD")
        if get_num("ev_spe"): ev_parts.append(f"{get_num('ev_spe')} Spe")
        if ev_parts:
            lines.append("EVs: " + " / ".join(ev_parts))

        if nature:
            lines.append(f"{nature} Nature")

        iv_parts = []
        for key, label in [
            ("iv_hp", "HP"),
            ("iv_atk", "Atk"),
            ("iv_def", "Def"),
            ("iv_spa", "SpA"),
            ("iv_spd", "SpD"),
            ("iv_spe", "Spe"),
        ]:
            v = get_iv(key)
            if v is not None:
                iv_parts.append(f"{v} {label}")
        if iv_parts:
            lines.append("IVs: " + " / ".join(iv_parts))

    for mv in [row.get("move1"), row.get("move2"), row.get("move3"), row.get("move4")]:
        mv = (mv or "").strip()
        if mv:
            lines.append(f"- {mv}")

    return "\n".join(lines).strip()

def summarize_row(row: dict) -> str:
    species = (row.get("pokemon") or row.get("species") or "").strip()
    nickname = (row.get("nickname") or "").strip()
    item = (row.get("item") or "").strip()
    ability = (row.get("ability") or "").strip()
    level = row.get("level")

    title = f"{nickname} ({species})" if nickname else species
    extra = []
    if level not in (None, "", 0, "0"):
        extra.append(f"Lv {level}")
    if ability:
        extra.append(ability)
    if item:
        extra.append(f"@ {item}")
    if row.get("shiny", False):
        extra.append("⭐")

    return f"• **{title}**" + ((" — " + ", ".join(extra)) if extra else "")

def make_pokepaste_file(rows: List[dict], ots: bool, api_base: str) -> discord.File:
    blocks = [build_pokepaste(r, ots=ots) for r in rows]
    all_text = "\n\n".join(blocks).strip()
    header = f"Mode: {'OTS' if ots else 'Full'} | API: {api_base} | Count: {len(rows)}"
    pokepaste_text = header + "\n\n" + all_text + "\n"
    fp = io.BytesIO(pokepaste_text.encode("utf-8"))
    return discord.File(fp, filename="pokepaste.txt")


# -----------------------------
# HTTP (API calls)
# -----------------------------
class ApiClient:
    def __init__(self, base: str):
        self.base = base

    async def parse_images(self, session: aiohttp.ClientSession, files: List[Tuple[str, bytes]]) -> dict:
        url = self.base + "/parse"
        data = aiohttp.FormData()
        for filename, content in files:
            data.add_field("files", content, filename=filename, content_type="application/octet-stream")

        async with session.post(url, data=data) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"API /parse failed: {resp.status} {text[:500]}")
            return await resp.json()

    async def parse_video(self, session: aiohttp.ClientSession, filename: str, content: bytes) -> dict:
        url = self.base + "/parse_video"
        data = aiohttp.FormData()
        data.add_field("file", content, filename=filename, content_type="application/octet-stream")

        async with session.post(url, data=data) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"API /parse_video failed: {resp.status} {text[:500]}")
            return await resp.json()

# -----------------------------
# Discord UI (Buttons + Select)
# -----------------------------
class ResultView(discord.ui.View):
    def __init__(self, rows: List[dict], ots: bool, timeout: float = 15 * 60):
        super().__init__(timeout=timeout)
        self.rows = rows
        self.ots = ots

        # populate select with up to 25 (Discord limit per select)
        opts = []
        for idx, r in enumerate(rows[:25]):
            species = (r.get("pokemon") or r.get("species") or "").strip() or "Unknown"
            nick = (r.get("nickname") or "").strip()
            label = f"{idx+1}. {nick} ({species})" if nick else f"{idx+1}. {species}"
            # label max 100 chars
            label = label[:100]
            desc_bits = []
            lvl = r.get("level")
            if lvl not in (None, "", 0, "0"):
                desc_bits.append(f"Lv {lvl}")
            ab = (r.get("ability") or "").strip()
            if ab:
                desc_bits.append(ab)
            desc = ", ".join(desc_bits)[:100] if desc_bits else None

            opts.append(discord.SelectOption(label=label, description=desc, value=str(idx)))

        if opts:
            self.add_item(PokemonSelect(opts, self))

    def current_mode_label(self) -> str:
        return "OTS" if self.ots else "Full"

    @discord.ui.button(label="Toggle OTS (currently: Full)", style=discord.ButtonStyle.secondary, custom_id="toggle_ots")
    async def toggle_ots(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.ots = not self.ots
        button.label = f"Toggle OTS (currently: {self.current_mode_label()})"
        await interaction.response.send_message(
            f"Mode is now **{self.current_mode_label()}**. (New copies will respect this.)",
            ephemeral=True
        )

    @discord.ui.button(label="Copy ALL PokePaste", style=discord.ButtonStyle.primary, custom_id="copy_all")
    async def copy_all(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Use ephemeral file so it’s “copyable” without spamming channel
        blocks = [build_pokepaste(r, ots=self.ots) for r in self.rows]
        text = "\n\n".join(blocks).strip() + "\n"
        fp = io.BytesIO(text.encode("utf-8"))
        await interaction.response.send_message(
            content=f"Here’s **ALL PokePaste** (Mode: {self.current_mode_label()})",
            file=discord.File(fp, filename="pokepaste_all.txt"),
            ephemeral=True
        )

    @discord.ui.button(label="Copy Selected PokePaste", style=discord.ButtonStyle.success, custom_id="copy_selected")
    async def copy_selected(self, interaction: discord.Interaction, button: discord.ui.Button):
        # The select stores the last chosen index on the view
        idx = getattr(self, "selected_idx", None)
        if idx is None:
            return await interaction.response.send_message("Pick a Pokémon from the dropdown first.", ephemeral=True)

        r = self.rows[idx]
        text = build_pokepaste(r, ots=self.ots).strip() + "\n"
        fp = io.BytesIO(text.encode("utf-8"))
        await interaction.response.send_message(
            content=f"Selected PokePaste (Mode: {self.current_mode_label()})",
            file=discord.File(fp, filename="pokepaste_selected.txt"),
            ephemeral=True
        )

    async def on_error(self, interaction: discord.Interaction, error: Exception, item):
        try:
            await interaction.response.send_message(
                f"❌ UI error: `{type(error).__name__}: {error}`",
                ephemeral=True
            )
        except discord.InteractionResponded:
            await interaction.followup.send(
                f"❌ UI error: `{type(error).__name__}: {error}`",
                ephemeral=True
            )

class PokemonSelect(discord.ui.Select):
    def __init__(self, options: List[discord.SelectOption], view_ref: ResultView):
        super().__init__(
            placeholder="Select a Pokémon…",
            min_values=1,
            max_values=1,
            options=options,
            custom_id="pokemon_select"
        )
        self.view_ref = view_ref

    async def callback(self, interaction: discord.Interaction):
        idx = int(self.values[0])
        self.view_ref.selected_idx = idx

        r = self.view_ref.rows[idx]
        summary = summarize_row(r)
        await interaction.response.send_message(
            f"Selected: {summary}\nMode: **{self.view_ref.current_mode_label()}**",
            ephemeral=True
        )

# -----------------------------
# Discord bot
# -----------------------------
intents = discord.Intents.default()

class MyClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.api = ApiClient(API_BASE)

    async def setup_hook(self):
        await self.tree.sync()

client = MyClient()

MODE_CHOICES = [
    app_commands.Choice(name="Full (includes EV/IV/Nature)", value="full"),
    app_commands.Choice(name="OTS (hides EV/IV/Nature)", value="ots"),
]

def collect_attachments(*atts: Optional[discord.Attachment]) -> List[discord.Attachment]:
    return [a for a in atts if a is not None]


async def run_ocr_from_attachments(
    interaction: discord.Interaction,
    mode_value: Optional[str],
    attachments: List[discord.Attachment],
) -> Tuple[List[dict], bool]:
    ots = (mode_value == "ots") if mode_value else (DEFAULT_MODE == "ots")

    atts = list(attachments or [])
    if not atts:
        raise ValueError("Attach screenshots (images) or one video.")

    if len(atts) > MAX_ATTACHMENTS:
        raise ValueError(f"Too many attachments. Max is {MAX_ATTACHMENTS}.")

    images = [a for a in atts if is_image(a)]
    videos = [a for a in atts if is_video(a)]
    other = [a for a in atts if (a not in images and a not in videos)]

    if other:
        raise ValueError("Unsupported attachment type(s). Use only images or one video.")
    if images and videos:
        raise ValueError("Please attach either images OR one video, not both.")
    if videos and len(videos) > 1:
        raise ValueError("Please attach only one video at a time.")

    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if videos:
            v = videos[0]
            content = await v.read()
            data = await client.api.parse_video(session, v.filename, content)
            rows = data.get("rows") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                rows = []
            return rows, ots

        # images
        payload: List[Tuple[str, bytes]] = []
        for a in images:
            payload.append((a.filename, await a.read()))
        data = await client.api.parse_images(session, payload)
        rows = data.get("rows") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            rows = []
        return rows, ots


@client.tree.command(
    name="ocr",
    description="OCR screenshots (or one video). Returns PokePaste + interactive picker (no JSON)."
)
@app_commands.describe(
    mode="Full includes EV/IV/Nature. OTS hides them in PokePaste.",
    file1="Attachment 1",
    file2="Attachment 2",
    file3="Attachment 3",
    file4="Attachment 4",
    file5="Attachment 5",
    file6="Attachment 6",
    file7="Attachment 7",
    file8="Attachment 8",
    file9="Attachment 9",
    file10="Attachment 10",
)
@app_commands.choices(mode=MODE_CHOICES)
async def ocr(
    interaction: discord.Interaction,
    mode: Optional[app_commands.Choice[str]] = None,
    file1: Optional[discord.Attachment] = None,
    file2: Optional[discord.Attachment] = None,
    file3: Optional[discord.Attachment] = None,
    file4: Optional[discord.Attachment] = None,
    file5: Optional[discord.Attachment] = None,
    file6: Optional[discord.Attachment] = None,
    file7: Optional[discord.Attachment] = None,
    file8: Optional[discord.Attachment] = None,
    file9: Optional[discord.Attachment] = None,
    file10: Optional[discord.Attachment] = None,
):
    await interaction.response.defer(thinking=True)

    attachments = collect_attachments(
        file1, file2, file3, file4, file5, file6, file7, file8, file9, file10
    )

    try:
        rows, ots = await run_ocr_from_attachments(
            interaction,
            mode.value if mode else None,
            attachments
        )
    except Exception as e:
        return await interaction.followup.send(f"❌ {str(e)}", ephemeral=True)

    if not rows:
        return await interaction.followup.send(
            "No Pokémon detected. Try a clearer screenshot (default theme helps).",
            ephemeral=True
        )

    summaries = "\n".join(summarize_row(r) for r in rows[:25])
    if len(rows) > 25:
        summaries += f"\n… and **{len(rows)-25}** more."

    # Build view BEFORE sending
    view = ResultView(rows, ots=ots)

    # Set correct initial label on toggle button
    for item in view.children:
        if isinstance(item, discord.ui.Button) and item.custom_id == "toggle_ots":
            item.label = f"Toggle OTS (currently: {'OTS' if ots else 'Full'})"

    # Only attach PokePaste (no JSON)
    f_pokepaste = make_pokepaste_file(rows, ots=ots, api_base=API_BASE)

    await interaction.followup.send(
        content=(
            f"✅ Done! Found **{len(rows)}** Pokémon.\n"
            f"Mode: **{'OTS' if ots else 'Full'}** | API: `{API_BASE}`\n\n"
            f"{summaries}\n\n"
            f"Use the dropdown + buttons to copy **ALL** or **selected** PokePaste."
        ),
        file=f_pokepaste,
        view=view
    )


@client.tree.command(name="ocr_json", description="OCR and return ONLY JSON (rows.json).")
@app_commands.describe(
    mode="Full/OTS doesn't change JSON; it only matters for PokePaste elsewhere.",
    attachment1="Upload a screenshot (or a video).",
    attachment2="Optional extra screenshot.",
    attachment3="Optional extra screenshot.",
    attachment4="Optional extra screenshot.",
    attachment5="Optional extra screenshot.",
    attachment6="Optional extra screenshot.",
    attachment7="Optional extra screenshot.",
    attachment8="Optional extra screenshot.",
    attachment9="Optional extra screenshot.",
    attachment10="Optional extra screenshot.",
)
@app_commands.choices(mode=MODE_CHOICES)
async def ocr_json(
    interaction: discord.Interaction,
    mode: Optional[app_commands.Choice[str]] = None,
    attachment1: Optional[discord.Attachment] = None,
    attachment2: Optional[discord.Attachment] = None,
    attachment3: Optional[discord.Attachment] = None,
    attachment4: Optional[discord.Attachment] = None,
    attachment5: Optional[discord.Attachment] = None,
    attachment6: Optional[discord.Attachment] = None,
    attachment7: Optional[discord.Attachment] = None,
    attachment8: Optional[discord.Attachment] = None,
    attachment9: Optional[discord.Attachment] = None,
    attachment10: Optional[discord.Attachment] = None,
):
    await interaction.response.defer(thinking=True)

    attachments = collect_attachments(attachment1, attachment2, attachment3, attachment4, attachment5, attachment6, attachment7, attachment8, attachment9, attachment10)

    try:
        rows, ots = await run_ocr_from_attachments(interaction, mode.value if mode else None, attachments)
    except Exception as e:
        return await interaction.followup.send(f"❌ {str(e)}", ephemeral=True)

    if not rows:
        return await interaction.followup.send("No Pokémon detected.", ephemeral=True)

    normalized_rows = [normalize_pokemon_row(r) for r in rows]
    json_bytes = json.dumps(normalized_rows, ensure_ascii=False, indent=2).encode("utf-8")
    fp = io.BytesIO(json_bytes)

    await interaction.followup.send(
        content=f"✅ JSON ready. Count: **{len(normalized_rows)}** | API: `{API_BASE}`",
        file=discord.File(fp, filename="rows.json")
    )

@client.tree.command(name="ocr_pokepaste", description="OCR and return ONLY PokePaste (pokepaste.txt).")
@app_commands.describe(
    mode="Full includes EV/IV/Nature. OTS hides them.",
    attachment1="Upload a screenshot (or a video).",
    attachment2="Optional extra screenshot.",
    attachment3="Optional extra screenshot.",
    attachment4="Optional extra screenshot.",
    attachment5="Optional extra screenshot.",
    attachment6="Optional extra screenshot.",
    attachment7="Optional extra screenshot.",
    attachment8="Optional extra screenshot.",
    attachment9="Optional extra screenshot.",
    attachment10="Optional extra screenshot.",
)
@app_commands.choices(mode=MODE_CHOICES)
async def ocr_pokepaste(
    interaction: discord.Interaction,
    mode: Optional[app_commands.Choice[str]] = None,
    attachment1: Optional[discord.Attachment] = None,
    attachment2: Optional[discord.Attachment] = None,
    attachment3: Optional[discord.Attachment] = None,
    attachment4: Optional[discord.Attachment] = None,
    attachment5: Optional[discord.Attachment] = None,
    attachment6: Optional[discord.Attachment] = None,
    attachment7: Optional[discord.Attachment] = None,
    attachment8: Optional[discord.Attachment] = None,
    attachment9: Optional[discord.Attachment] = None,
    attachment10: Optional[discord.Attachment] = None,
):
    await interaction.response.defer(thinking=True)

    attachments = collect_attachments(attachment1, attachment2, attachment3, attachment4, attachment5, attachment6, attachment7, attachment8, attachment9, attachment10)

    try:
        rows, ots = await run_ocr_from_attachments(interaction, mode.value if mode else None, attachments)
    except Exception as e:
        return await interaction.followup.send(f"❌ {str(e)}", ephemeral=True)

    if not rows:
        return await interaction.followup.send("No Pokémon detected.", ephemeral=True)

    blocks = [build_pokepaste(r, ots=ots) for r in rows]
    text = "\n\n".join(blocks).strip() + "\n"
    fp = io.BytesIO(text.encode("utf-8"))

    await interaction.followup.send(
        content=f"✅ PokePaste ready. Mode: **{'OTS' if ots else 'Full'}** | Count: **{len(rows)}** | API: `{API_BASE}`",
        file=discord.File(fp, filename="pokepaste.txt")
    )


@client.tree.command(name="ping", description="Check if the bot is alive.")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("pong ✅", ephemeral=True)

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
