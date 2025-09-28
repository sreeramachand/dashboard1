# app.py
import os
import io
import uuid
import base64
import threading
from PIL import Image
from PyPDF2 import PdfReader
import pandas as pd
from time import sleep

import dash
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
#import dash_design_kit as ddk

# OpenAI client
from openai import OpenAI
from dotenv import load_dotenv

# --------------------
# Configuration
# --------------------
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# Max upload size enforced on server logic (100 MB)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# How many recent messages we keep in the UI (not sent)
MAX_UI_HISTORY = 200

# Maximum characters for file summary (keeps token consumption low)
MAX_FILE_SUMMARY_CHARS = 400  # ~ <100 tokens in practice (rough estimate)

# Poll interval for streaming UI update (ms)
POLL_INTERVAL_MS = 400

# Global in-memory streams buffer (thread -> partial text)
# Background streaming threads will write here; poll callback reads it.
STREAM_BUFFERS = {}

# --------------------
# Helpers: file metadata extraction & light summarization
# --------------------
def safe_truncate(text: str, max_chars: int):
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."

def summarize_image_basic(image_bytes: bytes, filename: str):
    """Extract basic metadata: dimensions and succinct caption derived from filename."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            width, height = im.size
            mode = im.mode
        # Basic caption: use filename split + short phrase.
        name_hint = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
        caption = f"{name_hint}" if name_hint else "Image"
        summary = f"Image: {filename}, {width}x{height}, mode={mode}. Caption: {caption}."
    except Exception:
        summary = f"Image: {filename}. (could not read dimensions)"
    return safe_truncate(summary, MAX_FILE_SUMMARY_CHARS)

def summarize_csv_basic(csv_bytes: bytes, filename: str):
    """Read only the header and produce a concise metadata summary."""
    try:
        # Use pandas to read only headers / small sample
        s = io.StringIO(csv_bytes.decode("utf-8", errors="replace"))
        df = pd.read_csv(s, nrows=3)  # small sample to infer columns
        cols = list(df.columns)
        summary = f"CSV: {filename}. Columns: {', '.join(cols)}. Rows(sample): {len(df)}."
    except Exception:
        summary = f"CSV: {filename}. (could not parse columns)"
    return safe_truncate(summary, MAX_FILE_SUMMARY_CHARS)

def summarize_text_basic(text_bytes: bytes, filename: str):
    """Return a short excerpt as summary."""
    try:
        txt = text_bytes.decode("utf-8", errors="replace")
        first = txt.strip().replace("\n", " ")[: MAX_FILE_SUMMARY_CHARS]
        summary = f"Text file: {filename}. Excerpt: {first}"
    except Exception:
        summary = f"Text file: {filename} (could not read)"
    return safe_truncate(summary, MAX_FILE_SUMMARY_CHARS)

def summarize_pdf_basic(pdf_bytes: bytes, filename: str):
    """Extract first page text and produce a short summary/excerpt."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for i, p in enumerate(reader.pages[:2]):  # just first two pages
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            pages_text.append(t.strip().replace("\n", " "))
        excerpt = " ".join(pages_text)[: MAX_FILE_SUMMARY_CHARS]
        summary = f"PDF: {filename}. Excerpt: {excerpt}"
    except Exception:
        summary = f"PDF: {filename}. (could not extract text)"
    return safe_truncate(summary, MAX_FILE_SUMMARY_CHARS)

def detect_and_summarize_file(content_b64: str, filename: str):
    """
    Given a content string like 'data:...;base64,AAA...', and the filename,
    return a small metadata object:
      { filename, content_type, size_bytes, summary }
    This function deliberately avoids sending large content to OpenAI.
    """
    if not content_b64:
        return None

    # Split header and base64 payload
    if "," in content_b64:
        header, b64 = content_b64.split(",", 1)
    else:
        b64 = content_b64
        header = ""
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raw = b""

    size_bytes = len(raw)
    # Basic content type detection from filename extension
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    if size_bytes > MAX_UPLOAD_BYTES:
        # mark as too big, but still store metadata
        return {
            "filename": filename,
            "content_type": f"application/{ext}",
            "size_bytes": size_bytes,
            "summary": f"File {filename} exceeds max allowed size ({size_bytes} bytes)."
        }

    if ext in ["png", "jpg", "jpeg", "gif", "bmp", "webp"]:
        summary = summarize_image_basic(raw, filename)
        content_type = f"image/{ext}"
    elif ext in ["csv"]:
        summary = summarize_csv_basic(raw, filename)
        content_type = "text/csv"
    elif ext in ["txt", "md", "text"]:
        summary = summarize_text_basic(raw, filename)
        content_type = "text/plain"
    elif ext in ["pdf"]:
        summary = summarize_pdf_basic(raw, filename)
        content_type = "application/pdf"
    else:
        # unknown: just produce a tiny summary including filename and size
        summary = f"{filename} ({size_bytes} bytes). Unknown extension .{ext}."
        content_type = f"application/{ext}"

    # short canonical size string
    size_kb = size_bytes / 1024.0
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024.0:.1f} MB"

    metadata = {
        "filename": filename,
        "content_type": content_type,
        "size_bytes": size_bytes,
        "size_human": size_str,
        # The summary is short and designed to remain under ~100 tokens
        "summary": summary[: MAX_FILE_SUMMARY_CHARS]
    }
    return metadata

# --------------------
# Dash app
# --------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True, prevent_initial_callbacks='initial_duplicate')

# 🕓 History

app.layout = [
    html.Div(children="Prod Analytics Dashboard", style={"textAlign": 'center', "fontSize": 28, "padding": "12px"}),
    
    html.Div(
    [
    dcc.Tabs(id="tabs", value="tab-chat", children=[
        dcc.Tab(label="📊 Data", value="tab-data"),
        dcc.Tab(label="🤖 Chat", value="tab-chat"),
    ]),
    html.Div(id="tabs-content"),
])]

# Render tabs (single callback)
@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == "tab-data":
        return html.Div(
            [html.H3("📊 Data Analysis - coming soon"), html.P("Replace with your charts")],
            style={"padding": 20}
        )
    # Chat tab
    return html.Div([
        html.H3("🤖 Chat with AI ✨", className="my-3"),
        html.Div(id="chat-history",
                 style={
                     "height": "450px",
                     "overflowY": "auto",
                     "border": "1px solid #ccc",
                     "padding": "10px",
                     "borderRadius": "6px",
                     "display": "flex",
                     "flexDirection": "column",
                 }),
        dbc.InputGroup([
            dbc.Input(id="user-input", placeholder="Type your message...", type="text"),
            dbc.Button("Send", id="send-btn", color="primary", n_clicks=0),
        ], className="my-3"),
        dcc.Upload(
            id="file-upload",
            children=html.Div(["📎 Drag and Drop or ", html.A("Select Files")]),
            style={"width": "100%", "height": "60px", "lineHeight": "60px",
                   "borderWidth": "1px", "borderStyle": "dashed",
                   "borderRadius": "5px", "textAlign": "center", "margin": "10px"},
            multiple=True
        ),
        html.Div(id="file-list", style={"marginBottom": 10}),
        # Stores:
        # chat_store: list of UI messages (user/assistant) for rendering only.
        # chat_summary_store: a short summary of conversation (not sent to OpenAI).
        # file_metadata_store: list of file metadata objects created on upload
        # stream_store: holds the active stream id and status
        dcc.Store(id="chat-store", data=[]),
        dcc.Store(id="chat-summary-store", data=""),  # plain string summary (not sent)
        dcc.Store(id="file-metadata-store", data=[]),
        dcc.Store(id="stream-store", data={"stream_id": None, "in_progress": False}),
        dcc.Interval(id="poll-interval", interval=POLL_INTERVAL_MS, n_intervals=0)
    ], style={"padding": 15})

# --------------------
# File upload handler:
#   - extracts small metadata summaries for each file
#   - stores metadata in file-metadata-store
#   - file-list displays filenames & short summary
# --------------------
@app.callback(
    Output("file-list", "children"),
    Output("file-metadata-store", "data"),
    Input("file-upload", "contents"),
    State("file-upload", "filename"),
    prevent_initial_call=True
)
def handle_upload(contents, filenames):
    """
    contents: list of data URLs e.g. 'data:image/png;base64,AAA...'
    filenames: list of filenames
    """
    if not contents:
        return [], []

    file_meta = []
    display_items = []
    for content, fname in zip(contents, filenames):
        # quick check for oversize using base64 length -> bytes
        try:
            if "," in content:
                _, b64 = content.split(",", 1)
            else:
                b64 = content
            byte_len = (len(b64) * 3) // 4
        except Exception:
            byte_len = 0

        if byte_len > MAX_UPLOAD_BYTES:
            # Save only metadata denoting oversize
            meta = {
                "filename": fname,
                "size_bytes": byte_len,
                "size_human": f"{byte_len / (1024*1024):.1f} MB",
                "content_type": "unknown",
                "summary": f"{fname} is too large ({byte_len} bytes) to process (limit {MAX_UPLOAD_BYTES} bytes)."
            }
        else:
            meta = detect_and_summarize_file(content, fname)

        file_meta.append(meta)
        display_items.append(html.Div([
            html.Strong(fname), html.Span(f" — {meta.get('size_human','?')}"),
            html.Div(meta.get("summary", ""), style={"fontSize": 12, "color": "#666"})
        ], style={"padding": 6, "borderBottom": "1px solid #eee"}))

    return display_items, file_meta

# --------------------
# Start chat (user presses Enter or clicks Send)
#  - Append user message to chat-store (UI)
#  - Append assistant placeholder (empty string) to chat-store (UI) for streaming
#  - Create a stream_id and spawn a background thread that streams tokens into STREAM_BUFFERS
#  - Update stream-store with stream_id & in_progress flag
#  - Clear uploaded files (we keep metadata in file-metadata-store but clear upload UI state)
# --------------------
@app.callback(
    Output("chat-store", "data"),
    Output("chat-history", "children", allow_duplicate=True),
    Output("user-input", "value"),
    Output("file-metadata-store", "data", allow_duplicate=True),  # we clear upload metadata here; allow duplicate
    Output("stream-store", "data"),
    Input("send-btn", "n_clicks"),
    Input("user-input", "n_submit"),
    State("user-input", "value"),
    State("chat-store", "data"),
    State("file-metadata-store", "data"),
    State("chat-summary-store", "data"),
    prevent_initial_call=True
)
def start_chat(n_clicks, n_submit, user_msg, chat_history, file_metadata, chat_summary):
    """
    We:
      - create an entry for the user's message (UI)
      - create a blank assistant entry for streaming to fill
      - spawn thread to call OpenAI with latest prompt + file metadata summaries
      - update chat_summary_store locally (not sent to OpenAI)
    """
    # Guard
    if not user_msg and not file_metadata:
        return chat_history, render_history(chat_history), "", file_metadata, {"stream_id": None, "in_progress": False}

    # Build the message shown in UI for user:
    user_text = user_msg or ""
    if file_metadata:
        # Append tiny summaries for user visibility in UI only
        for fm in file_metadata:
            user_text += f"\n\n[FILE] {fm['filename']}: {fm.get('summary','')}"
    user_entry = {"role": "user", "content": user_text}

    # Append to UI chat history
    chat_history = chat_history or []
    chat_history.append(user_entry)
    # Add assistant placeholder entry for streaming
    chat_history.append({"role": "assistant", "content": ""})

    # Update local chat summary (never sent to OpenAI by default)
    # We keep this short: include the user intent + file names + small hint
    summary_parts = []
    summary_parts.append(f"User: {safe_truncate(user_msg or '', 300)}")
    if file_metadata:
        fparts = [f"{fm['filename']} ({fm.get('size_human','?')})" for fm in file_metadata]
        summary_parts.append("Files: " + ", ".join(fparts))
    # Append to existing summary (keep it short)
    new_summary = chat_summary or ""
    addition = " | ".join(summary_parts)
    if new_summary:
        new_summary = safe_truncate(new_summary + " ; " + addition, 1500)
    else:
        new_summary = safe_truncate(addition, 1500)

    # Prepare concise system message + user prompt for OpenAI:
    # We only send:
    #  - a short system role
    #  - a small list of file metadata summaries (each < MAX_FILE_SUMMARY_CHARS)
    #  - the user's latest message (no chat history)
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Use only the provided file summaries to answer the user's question. Do not assume additional context."
    }

    # Build concise file summaries payload (strings)
    file_summaries = []
    if file_metadata:
        for fm in file_metadata:
            file_summaries.append(f"- {fm['filename']}: {safe_truncate(fm.get('summary',''), 180)}")

    file_summaries_text = "\n".join(file_summaries) if file_summaries else "No files uploaded."

    user_prompt_for_api = {
        "role": "user",
        "content": f"Files summary:\n{file_summaries_text}\n\nUser question:\n{user_msg or ''}"
    }

    messages_for_api = [system_msg, user_prompt_for_api]

    # Create a unique stream id
    stream_id = str(uuid.uuid4())
    STREAM_BUFFERS[stream_id] = {"text": "", "in_progress": True}

    # Background thread: call OpenAI streaming and write incremental tokens into STREAM_BUFFERS[stream_id]["text"]
    def background_stream(stream_id, messages):
        buffer_obj = STREAM_BUFFERS.get(stream_id)
        try:
            # Call OpenAI with streaming=True
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True
            )
            # Iterate events — the SDK yields "delta" chunks
            partial = ""
            for event in response:
                # Some events may not carry content
                try:
                    delta = event.choices[0].delta
                except Exception:
                    delta = None
                # delta may have content in 'content' or 'content' nested; handle safely
                chunk = ""
                if delta:
                    # delta could be a dict-like with 'content'
                    chunk = getattr(delta, "content", None) or delta.get("content") if isinstance(delta, dict) else ""
                    if not chunk:
                        # older SDK shape: delta is e.g. {"role":"assistant"} or {"content":"..."}
                        chunk = delta.get("content") if isinstance(delta, dict) else ""
                if chunk:
                    partial += chunk
                    # update shared buffer
                    buffer_obj["text"] = partial
                # tiny sleep to yield
                # sleep(0.001)
            # Finished
            buffer_obj["in_progress"] = False
        except Exception as e:
            # On error put message into buffer
            buffer_obj["text"] += f"\n\n[Stream error: {repr(e)}]"
            buffer_obj["in_progress"] = False

    # Start background streaming thread
    t = threading.Thread(target=background_stream, args=(stream_id, messages_for_api), daemon=True)
    t.start()

    # stream-store status to be returned to front-end (used by polling)
    stream_state = {"stream_id": stream_id, "in_progress": True}

    # Clear file metadata store (we already stored the file metadata in file-metadata-store - keep copy but here we clear to avoid re-sending)
    # NOTE: We return allow_duplicate=True for this output so the decorator doesn't conflict with upload callback.
    cleared_file_metadata = []

    # Return values:
    #  chat-store (data), rendered chat-history children, cleared input value, cleared file metadata, stream-store
    return chat_history, render_history(chat_history), "", cleared_file_metadata, stream_state

# --------------------
# Polling callback: read STREAM_BUFFERS for active stream and update assistant bubble progressively
# We write chat-history.children and update chat-store.data with the latest assistant text.
# chat-store.data is also written by start_chat, so we set allow_duplicate=True here.
# --------------------
@app.callback(
    Output("chat-history", "children"),
    Output("chat-store", "data", allow_duplicate=True),
    Input("poll-interval", "n_intervals"),
    State("chat-store", "data"),
    State("stream-store", "data"),
    prevent_initial_call=False
)
def poll_stream_and_update(n_intervals, chat_history, stream_store):
    """
    Poll STREAM_BUFFERS for the current stream_id. If in_progress, update UI
    by replacing last assistant message content with the partial text.
    """
    # If no chat history, nothing to render
    if not chat_history:
        return [], chat_history or []

    # If no active stream, just render
    stream_id = stream_store.get("stream_id") if stream_store else None
    if not stream_id or stream_id not in STREAM_BUFFERS:
        return render_history(chat_history), chat_history

    buf = STREAM_BUFFERS.get(stream_id, {"text": "", "in_progress": False})
    partial_text = buf.get("text", "")
    in_progress = buf.get("in_progress", False)

    # Update the last assistant message content (it exists because we appended a placeholder earlier)
    # Make sure last message is assistant - otherwise avoid overwriting
    if chat_history and chat_history[-1].get("role") == "assistant":
        chat_history[-1]["content"] = partial_text

    # If streaming finished, and buffer indicates finished, we can optionally finalize:
    if not in_progress:
        # stream complete. remove buffer to free memory
        try:
            del STREAM_BUFFERS[stream_id]
        except Exception:
            pass
        # mark stream_store as finished by returning stream_store with in_progress False (frontend can update)
        # Note: poll_stream does not directly update stream-store here (it is only read). If you want to clear it, you can do so via other callback.
    # Render and return
    return render_history(chat_history), chat_history

# --------------------
# Utility: Render chat history to UI elements
# --------------------
def render_history(history):
    """Render the list of messages into styled chat bubbles (user right, assistant left)."""
    items = []
    base_style = {
        "margin": "6px",
        "padding": "10px",
        "borderRadius": "10px",
        "maxWidth": "85%",
        "whiteSpace": "pre-wrap",
    }
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            style = {**base_style, "backgroundColor": "#DCF8C6", "alignSelf": "flex-end", "textAlign": "right"}
            items.append(html.Div(content, style=style))
        else:
            style = {**base_style, "backgroundColor": "#fff", "border": "1px solid #eee",
                     "alignSelf": "flex-start", "textAlign": "left"}
            # Use Markdown for assistant if you'd like; we use simple text rendering here.
            # If you want Markdown, replace html.Div(content, ...) with:
            # html.Div(dcc.Markdown(content, link_target="_blank"), style=style)
            items.append(html.Div(dcc.Markdown(content, link_target="_blank") if content else html.Div(""), style=style))
    return items

# --------------------
# Optional: endpoint to retrieve the saved chat summary (not sent to OpenAI)
# --------------------
@app.callback(
    Output("chat-summary-store", "data"),
    Input("chat-store", "data"),
    State("chat-summary-store", "data"),
    prevent_initial_call=False
)
def maintain_chat_summary(chat_store, current_summary):
    """
    This callback keeps a trimmed conversational summary in memory.
    For minimal token consumption, we generate a tiny summary locally:
      - Append the last user line (truncated) and file names, not the whole content.
    The summary is stored in chat-summary-store and NEVER automatically sent to OpenAI.
    """
    if not chat_store:
        return current_summary or ""

    # find last user message and last file mentions (if any)
    last_user = None
    for m in reversed(chat_store):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break

    if not last_user:
        return current_summary or ""

    addition = safe_truncate(last_user.replace("\n", " "), 300)
    if current_summary:
        new_summary = safe_truncate(current_summary + " ; " + addition, 1500)
    else:
        new_summary = addition
    return new_summary

# --------------------
# Run server
# --------------------
if __name__ == "__main__":
    app.run(debug=True)