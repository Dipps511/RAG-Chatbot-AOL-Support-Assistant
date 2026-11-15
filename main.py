"""
========================================================================================
Art of Living (AOL) Wisdom Support - AI Volunteer "Ananda"
========================================================================================
A Streamlit web application providing intelligent, warm, and personalized support for 
The Art of Living wisdom, meditation guidance, course recommendations, and local 
teacher/program directories.

Architecture Overview:
1. Retrieval-Augmented Generation (RAG):
   - Retrieves authoritative wisdom chunks from AWS Bedrock Knowledge Base.
   - Synthesizes answers using Claude (Anthropic via AWS Bedrock Runtime).
2. Inline Citation & Hover Previews:
   - Replaces numbered citation markers [1], [2] with interactive hover popovers.
   - Enables scrolling excerpts and full-sheet reading modals without losing context.
3. Personalization & Local Connection:
   - Onboarding interactive quick-chips detect practice background (Sudarshan Kriya).
   - Automatically detects user location (e.g. Toronto) and connects them with local
     teachers (name, phone, email) and upcoming course schedules (next 2-4 weeks).
========================================================================================
"""

import datetime
import html
import json
import os
import re
from urllib.parse import quote_plus, urlparse
from zoneinfo import ZoneInfo

import boto3
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ======================================================================================
# 1. AWS BEDROCK CONFIGURATION & CLIENT INITIALIZATION
# ======================================================================================

# Helper to resolve configuration from Streamlit secrets (Cloud) or environment variables (Local)
def _get_config(key: str, default: str | None = None) -> str | None:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

REGION = _get_config("AWS_REGION", _get_config("AWS_DEFAULT_REGION", "us-east-2"))

# ARN / Model ID for Anthropic Claude on AWS Bedrock (used for answer generation)
MODEL_ID = _get_config("BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-5-20251101-v1:0")

# AWS Bedrock Knowledge Base ID containing indexed Art of Living wisdom documents/articles
KNOWLEDGE_BASE_ID = _get_config("BEDROCK_KB_ID", "GMBZWNP2XE")

accept = "application/json"
contentType = "application/json"

# Build boto3 client arguments (supports AWS credentials from Streamlit Secrets or ~/.aws)
_boto_kwargs = {"region_name": REGION}
_aws_access_key = _get_config("AWS_ACCESS_KEY_ID")
_aws_secret_key = _get_config("AWS_SECRET_ACCESS_KEY")
_aws_session_token = _get_config("AWS_SESSION_TOKEN")

if _aws_access_key and _aws_secret_key:
    _boto_kwargs["aws_access_key_id"] = _aws_access_key
    _boto_kwargs["aws_secret_access_key"] = _aws_secret_key
    if _aws_session_token:
        _boto_kwargs["aws_session_token"] = _aws_session_token

# Bedrock Agent Runtime client: handles vector search retrieval (`retrieve` API)
bedrock_agent_client = boto3.client("bedrock-agent-runtime", **_boto_kwargs)

# Bedrock Runtime client: handles LLM inference (`invoke_model` API with Claude)
bedrock_client = boto3.client(service_name="bedrock-runtime", **_boto_kwargs)

# ======================================================================================
# 2. STREAMLIT PAGE CONFIGURATION & CUSTOM STYLES
# ======================================================================================

st.set_page_config(
    page_title="Art of Living Wisdom Support",
    page_icon="https://cdn.prod.website-files.com/6a5fae7925920274d51d0c91/6a698d55f3efcdb0992e23c8_logo.svg",
    layout="wide",
)

# Custom CSS for the UI:
# - Fixes container clipping so hover popovers can float freely outside the chat bubbles.
# - Renders citation badges ([1], [2]) with an invisible hover bridge and a 450ms closing grace
#   delay to ensure smooth mouse transit from badge to floating preview.
st.markdown(
    """
<style>
/* Prevent message container clipping hover popups */
.stChatMessage,
[data-testid="stChatMessageContent"],
[data-testid="stMarkdownContainer"],
.element-container {
    overflow: visible !important;
}

/* Inline Citation Pill with Hover Preview */
.cite-wrapper {
    position: relative;
    display: inline-block;
    vertical-align: baseline;
    margin: 0 3px;
}

/* Wide invisible hover bridge between the [1] badge and the popup window */
.cite-wrapper:hover::after {
    content: "";
    position: absolute;
    bottom: 100%;
    left: -50px;
    right: -50px;
    height: 16px;
    z-index: 999998;
}

.cite-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #e0e7ff;
    color: #3730a3 !important;
    border: 1px solid #c7d2fe;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    line-height: 1;
    padding: 2px 6px;
    text-decoration: none !important;
    cursor: pointer;
    transition: all 0.15s ease-in-out;
}

.cite-badge:hover {
    background: #4338ca;
    color: #ffffff !important;
    border-color: #3730a3;
    box-shadow: 0 2px 6px rgba(67, 56, 202, 0.35);
}

/* Hover popup window:
   - Uses opacity & visibility with transition-delay so it stays open for 450ms when moving the cursor.
   - Supports live resizing (resize: both) and vertical scrolling. */
.cite-popover {
    display: flex;
    visibility: hidden;
    opacity: 0;
    pointer-events: none;
    position: absolute;
    bottom: calc(100% + 4px);
    left: 50%;
    transform: translateX(-50%);
    width: 390px;
    min-width: 280px;
    max-width: 85vw;
    height: 230px;
    min-height: 140px;
    max-height: 60vh;
    background: #ffffff;
    color: #1f2937;
    border: 1px solid #c7d2fe;
    border-radius: 10px;
    box-shadow: 0 14px 34px rgba(30, 27, 75, 0.22);
    z-index: 999999;
    resize: both;
    overflow: hidden;
    flex-direction: column;
    transition: opacity 0.2s ease 0.45s, visibility 0.2s ease 0.45s;
}

.cite-wrapper:hover .cite-popover,
.cite-popover:hover {
    visibility: visible;
    opacity: 1;
    pointer-events: auto;
    transition-delay: 0s;
}

.cite-header {
    background: #eef2ff;
    border-bottom: 1px solid #e0e7ff;
    padding: 7px 11px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    font-weight: 700;
    color: #312e81;
    flex-shrink: 0;
}

.cite-hint {
    font-size: 0.70rem;
    font-weight: 400;
    color: #6366f1;
}

.cite-body {
    padding: 9px 12px;
    overflow-y: auto;
    font-size: 0.86rem;
    line-height: 1.5;
    color: #374151;
    white-space: pre-wrap;
    flex: 1 1 auto;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("AOL Wisdom Support")

# ======================================================================================
# 3. KNOWLEDGE SHEET FETCHING, PARSING & CACHING
# ======================================================================================

def _title_from_url(url: str, index: int) -> str:
    """
    Derive a clean, human-readable title from a URL path slug.
    Example: 'https://.../mend-broken-heart/' -> 'Mend Broken Heart'
    """
    if not url:
        return f"Knowledge Sheet {index}"
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    if path_parts:
        slug = re.sub(r"\.[a-zA-Z0-9]+$", "", path_parts[-1])
        cleaned = re.sub(r"[-_]+", " ", slug).strip()
        if cleaned:
            return cleaned.title()
    return f"Knowledge Sheet {index}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_full_knowledge_sheet(url: str, fallback_text: str) -> tuple[str, str]:
    """
    Scrapes the full text of an article from its web URL so users can read the
    entire knowledge sheet word-by-word. Results are cached for 1 hour.
    If the web scrape fails or the source is an S3 URI, it returns the fallback text (KB chunks).
    """
    if not url or not url.startswith(("http://", "https://")):
        return "", fallback_text

    try:
        resp = requests.get(
            url,
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AOLWisdomSupport/1.0)"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract the page title (h1 or title tag)
        h1 = soup.find("h1")
        page_title = h1.get_text(" ", strip=True) if h1 else ""
        if not page_title and soup.title:
            page_title = soup.title.get_text(" ", strip=True)

        # Target the main article body and strip away navigation, footers, and scripts
        container = (
            soup.select_one("article .entry-content")
            or soup.select_one(".entry-content")
            or soup.select_one(".post-content")
            or soup.find("article")
            or soup.find("main")
            or soup.body
        )

        if container:
            for tag in container.select(
                "script, style, nav, footer, aside, .sharedaddy, .related-posts"
            ):
                tag.decompose()

            paragraphs = []
            for el in container.find_all(["h2", "h3", "h4", "p", "li", "blockquote"]):
                if el.name == "p" and el.find_parent(["li", "blockquote"]):
                    continue
                text = el.get_text(" ", strip=True)
                if not text:
                    continue
                if el.name in ("h2", "h3", "h4"):
                    paragraphs.append(f"### {text}")
                elif el.name == "blockquote":
                    paragraphs.append(f"> {text}")
                elif el.name == "li":
                    paragraphs.append(f"- {text}")
                else:
                    paragraphs.append(text)

            full_text = "\n\n".join(paragraphs).strip()
            # Verify parsed text is substantial before replacing fallback chunks
            if len(full_text) >= len(fallback_text) * 0.6 and full_text:
                return page_title, full_text
    except Exception:
        pass

    return "", fallback_text


def build_knowledge_sheets(retrieval_results: list[dict]) -> list[dict]:
    """
    Aggregates raw retrieval chunks by source URL, assigns 1-based index numbers [1], [2]...,
    and fetches article titles and full article texts.
    """
    sheets = []
    seen_urls: dict[str, int] = {}

    for ref in retrieval_results:
        location = ref.get("location") or {}
        url = (
            (location.get("webLocation") or {}).get("url")
            or (location.get("s3Location") or {}).get("uri")
            or ""
        )
        chunk_text = ((ref.get("content") or {}).get("text") or "").strip()

        # If multiple chunks came from the same URL, merge them together
        if url and url in seen_urls:
            idx = seen_urls[url]
            if chunk_text and chunk_text not in sheets[idx]["chunks"]:
                sheets[idx]["chunks"].append(chunk_text)
            continue

        sheet_index = len(sheets) + 1
        seen_urls[url] = len(sheets)
        sheets.append(
            {
                "index": sheet_index,
                "url": url,
                "chunks": [chunk_text] if chunk_text else [],
            }
        )

    # Resolve readable titles and full article text for each sheet
    for sheet in sheets:
        combined_chunks = "\n\n---\n\n".join(sheet["chunks"]).strip()
        fetched_title, full_text = fetch_full_knowledge_sheet(sheet["url"], combined_chunks)
        sheet["title"] = fetched_title or _title_from_url(sheet["url"], sheet["index"])
        sheet["full_text"] = full_text or combined_chunks
        sheet["preview_text"] = full_text or combined_chunks or "Excerpt unavailable."

    return sheets


# ======================================================================================
# 4. MODALS & HOVER CITATION RENDERING
# ======================================================================================

@st.dialog("📖 Knowledge Sheet from Gurudev", width="large")
def show_knowledge_sheet_dialog(title: str, url: str, full_text: str):
    """
    Modal dialog providing a distraction-free, word-for-word reading experience
    of the complete Knowledge Sheet.
    """
    st.subheader(title)
    if url and url.startswith(("http://", "https://")):
        st.markdown(f"🔗 [Open original article on Wisdom Website]({url})")
    elif url:
        st.caption(f"Source: `{url}`")
    st.divider()
    with st.container(height=480):
        st.markdown(full_text)


def _format_inline_html(text: str) -> str:
    """Format markdown headings and quotes into clean HTML for display in the hover popover."""
    lines = []
    for line in text.splitlines():
        esc = html.escape(line)
        if esc.startswith("### "):
            lines.append(f"<strong>{esc[4:]}</strong>")
        elif esc.startswith("&gt; "):
            lines.append(f"<em>“{esc[5:]}”</em>")
        elif esc.startswith("- "):
            lines.append(f"• {esc[2:]}")
        else:
            lines.append(esc)
    return "<br>".join(lines)


def inject_hover_citations(answer_text: str, sheets: list[dict]) -> str:
    """
    Finds numbered citation tags like [1] or [2] in Claude's response text and replaces
    them with custom HTML badges that reveal scrollable hover popovers with excerpts.
    """
    if not sheets:
        return answer_text

    sheet_by_idx = {s["index"]: s for s in sheets}

    def _replace_cite(match: re.Match) -> str:
        idx_str = match.group(1)
        try:
            num = int(idx_str)
        except ValueError:
            return match.group(0)

        sheet = sheet_by_idx.get(num)
        if not sheet:
            return match.group(0)

        safe_title = html.escape(sheet["title"])
        safe_url = html.escape(sheet["url"] or "#")
        safe_body = _format_inline_html(sheet["preview_text"])

        return (
            f'<span class="cite-wrapper">'
            f'<a class="cite-badge" href="{safe_url}" target="_blank" rel="noopener noreferrer" title="Click to open source">[{num}]</a>'
            f'<span class="cite-popover">'
            f'<span class="cite-header">'
            f'<span>📖 [{num}] {safe_title}</span>'
            f'<span class="cite-hint">Scroll or drag ↘ to resize</span>'
            f'</span>'
            f'<span class="cite-body">{safe_body}</span>'
            f'</span>'
            f'</span>'
        )

    # Match bracketed digits [1], [2], etc.
    return re.sub(r"\[(\d+)\]", _replace_cite, answer_text)


def render_knowledge_sheets_bar(sheets: list[dict], msg_idx: int):
    """
    Renders reference buttons beneath each assistant response, allowing users to
    open popover excerpts or click 'Expand Full Screen' to open the reading modal.
    """
    if not sheets:
        return

    st.markdown("##### 📚 Referenced Knowledge Sheets")
    cols = st.columns(min(len(sheets), 3))
    for i, sheet in enumerate(sheets):
        with cols[i % len(cols)]:
            with st.popover(f"[{sheet['index']}] {sheet['title']}", use_container_width=True):
                st.markdown(f"### 📖 [{sheet['index']}] {sheet['title']}")
                if sheet["url"]:
                    st.markdown(f"🔗 [Open Original Article]({sheet['url']})")
                st.caption("Scroll through excerpt, or expand to read word-by-word:")
                with st.container(height=240):
                    st.markdown(sheet["preview_text"])
                if st.button(
                    "⤢ Expand Full Screen",
                    key=f"expand_btn_{msg_idx}_{i}",
                    use_container_width=True,
                ):
                    show_knowledge_sheet_dialog(
                        sheet["title"], sheet["url"], sheet["full_text"]
                    )


# ======================================================================================
# 5. LIVE LOCATION DETECTION & ART OF LIVING COURSE FINDER API INTEGRATION
# ======================================================================================

# Fast lookup table for common cities -> (latitude, longitude, country_code, display_label)
CITY_COORDINATES: dict[str, tuple[float, float, str, str]] = {
    "toronto": (43.6532, -79.3832, "ca", "Toronto, ON, Canada"),
    "mississauga": (43.5890, -79.6441, "ca", "Mississauga, ON, Canada"),
    "brampton": (43.7315, -79.7624, "ca", "Brampton, ON, Canada"),
    "vaughan": (43.8563, -79.5085, "ca", "Vaughan, ON, Canada"),
    "markham": (43.8561, -79.3370, "ca", "Markham, ON, Canada"),
    "scarborough": (43.7764, -79.2318, "ca", "Scarborough, ON, Canada"),
    "north york": (43.7615, -79.4111, "ca", "North York, ON, Canada"),
    "etobicoke": (43.6205, -79.5132, "ca", "Etobicoke, ON, Canada"),
    "oakville": (43.4675, -79.6877, "ca", "Oakville, ON, Canada"),
    "burlington": (43.3255, -79.7990, "ca", "Burlington, ON, Canada"),
    "hamilton": (43.2557, -79.8711, "ca", "Hamilton, ON, Canada"),
    "waterloo": (43.4643, -80.5204, "ca", "Waterloo, ON, Canada"),
    "kitchener": (43.4516, -80.4925, "ca", "Kitchener, ON, Canada"),
    "ottawa": (45.4215, -75.6972, "ca", "Ottawa, ON, Canada"),
    "montreal": (45.5017, -73.5673, "ca", "Montreal, QC, Canada"),
    "vancouver": (49.2827, -123.1207, "ca", "Vancouver, BC, Canada"),
    "surrey": (49.1913, -122.8490, "ca", "Surrey, BC, Canada"),
    "burnaby": (49.2488, -122.9805, "ca", "Burnaby, BC, Canada"),
    "calgary": (51.0447, -114.0719, "ca", "Calgary, AB, Canada"),
    "edmonton": (53.5461, -113.4938, "ca", "Edmonton, AB, Canada"),
    "winnipeg": (49.8951, -97.1384, "ca", "Winnipeg, MB, Canada"),
    "halifax": (44.6488, -63.5752, "ca", "Halifax, NS, Canada"),
    "new york": (40.7128, -74.0060, "us", "New York, NY, USA"),
    "nyc": (40.7128, -74.0060, "us", "New York, NY, USA"),
    "chicago": (41.8781, -87.6298, "us", "Chicago, IL, USA"),
    "los angeles": (34.0522, -118.2437, "us", "Los Angeles, CA, USA"),
    "san francisco": (37.7749, -122.4194, "us", "San Francisco, CA, USA"),
    "bay area": (37.3861, -122.0839, "us", "San Francisco Bay Area, CA, USA"),
    "san jose": (37.3382, -121.8863, "us", "San Jose, CA, USA"),
    "boston": (42.3601, -71.0589, "us", "Boston, MA, USA"),
    "seattle": (47.6062, -122.3321, "us", "Seattle, WA, USA"),
    "austin": (30.2672, -97.7431, "us", "Austin, TX, USA"),
    "dallas": (32.7767, -96.7970, "us", "Dallas, TX, USA"),
    "houston": (29.7604, -95.3698, "us", "Houston, TX, USA"),
    "atlanta": (33.7490, -84.3880, "us", "Atlanta, GA, USA"),
    "washington": (38.9072, -77.0369, "us", "Washington, DC, USA"),
    "miami": (25.7617, -80.1918, "us", "Miami, FL, USA"),
    "denver": (39.7392, -104.9903, "us", "Denver, CO, USA"),
    "detroit": (42.3314, -83.0458, "us", "Detroit, MI, USA"),
    "philadelphia": (39.9526, -75.1652, "us", "Philadelphia, PA, USA"),
    "phoenix": (33.4484, -112.0740, "us", "Phoenix, AZ, USA"),
    "london": (51.5072, -0.1276, "gb", "London, UK"),
    "mumbai": (19.0760, 72.8777, "in", "Mumbai, India"),
    "bengaluru": (12.9716, 77.5946, "in", "Bengaluru, India"),
    "bangalore": (12.9716, 77.5946, "in", "Bengaluru, India"),
    "delhi": (28.6139, 77.2090, "in", "New Delhi, India"),
    "new delhi": (28.6139, 77.2090, "in", "New Delhi, India"),
    "pune": (18.5204, 73.8567, "in", "Pune, India"),
    "hyderabad": (17.3850, 78.4867, "in", "Hyderabad, India"),
    "chennai": (13.0827, 80.2707, "in", "Chennai, India"),
    "sydney": (-33.8688, 151.2093, "au", "Sydney, Australia"),
    "melbourne": (-37.8136, 144.9631, "au", "Melbourne, Australia"),
}

COUNTRY_LANGUAGE_MAP = {
    "ca": "en-ca",
    "us": "en-us",
    "in": "en-in",
    "gb": "en-gb",
    "au": "en-au",
}


def detect_and_update_city(user_text: str, current_city: str | None) -> str | None:
    """
    Scans the user's input for a city name or location phrase (e.g., 'I live in Vancouver',
    'courses in Calgary', or a standalone city reply after Ananda asks for their city)
    and returns the detected city name.
    """
    cleaned = user_text.strip()
    lower_text = cleaned.lower()

    # 1. Check against common cities dictionary (longest match first)
    for city_key in sorted(CITY_COORDINATES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(city_key)}\b", lower_text):
            return "New York" if city_key == "nyc" else city_key.title()

    # 2. Regex patterns for explicit location statements ("I live in <City>", "courses in <City>", etc.)
    match = re.search(
        r"\b(?:i\s+live\s+in|i'm\s+in|i\s+am\s+in|located\s+in|based\s+in|from|courses\s+in|courses\s+near|programs\s+in|teachers\s+in|volunteer\s+in|city\s+is)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
        cleaned,
    )
    if match:
        candidate = match.group(1).strip()
        stop_words = {
            "Part", "Sahaj", "Sudarshan", "The", "Art", "Meditation", "Course",
            "Program", "Online", "Person", "Silence", "Kriya", "Gurudev", "English",
        }
        if candidate not in stop_words:
            return candidate

    # 3. If the user sends a short 1-3 word message that looks like a city reply (e.g. "Winnipeg" or "Halifax, NS")
    if current_city is None and 1 <= len(cleaned.split()) <= 3 and "?" not in cleaned:
        non_city_words = {
            "hi", "hello", "hey", "yes", "no", "thanks", "thank", "you", "ok", "okay",
            "sure", "help", "meditation", "breath", "kriya", "stress", "anxiety", "sleep",
            "courses", "course", "teacher", "teachers", "part", "sahaj", "samadhi",
        }
        tokens = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in cleaned.split()]
        tokens = [t for t in tokens if t]
        if tokens and not any(t in non_city_words for t in tokens):
            # Verify via geocoder if this short input is a valid city
            geo = geocode_city(cleaned)
            if geo is not None:
                return cleaned.rstrip(".,!")

    return current_city


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_city(city_query: str) -> tuple[float, float, str, str] | None:
    """
    Resolves a city name into (latitude, longitude, country_code, display_name).
    Checks the fast built-in table first, then falls back to OpenStreetMap Nominatim.
    """
    norm = city_query.strip().lower()
    if norm in CITY_COORDINATES:
        return CITY_COORDINATES[norm]

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": city_query.strip(),
                "format": "json",
                "addressdetails": 1,
                "limit": 1,
            },
            headers={"User-Agent": "AOL-Ananda-Support-Assistant/1.0"},
            timeout=6,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                item = data[0]
                lat = float(item["lat"])
                lng = float(item["lon"])
                country_code = (
                    item.get("address", {}).get("country_code", "ca").lower()
                )
                display_name = item.get("display_name", city_query.strip())
                return (lat, lng, country_code, display_name)
    except Exception:
        pass
    return None


def _format_course_schedule(course: dict) -> str:
    """
    Converts `bx_course_timings` Unix timestamps from the Art of Living Course API
    into a human-readable local date & session timing breakdown.
    """
    tz_name = course.get("field_time_zone") or course.get("timezone") or "America/Toronto"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = datetime.timezone.utc

    timings = course.get("bx_course_timings") or []
    if timings:
        day_entries = []
        for t in timings:
            start_ts = t.get("start_time")
            end_ts = t.get("end_time")
            if start_ts and end_ts:
                dt_start = datetime.datetime.fromtimestamp(int(start_ts), tz=tz)
                dt_end = datetime.datetime.fromtimestamp(int(end_ts), tz=tz)
                tz_abbr = dt_start.strftime("%Z")
                day_str = (
                    f"{dt_start.strftime('%a, %b %d, %Y')}: "
                    f"{dt_start.strftime('%I:%M %p').lstrip('0')} – "
                    f"{dt_end.strftime('%I:%M %p').lstrip('0')} {tz_abbr}"
                )
                day_entries.append(day_str)
        if day_entries:
            return " | ".join(day_entries)

    # Fallback to start_date and end_date strings if bx_course_timings is empty
    start_raw = (course.get("start_date") or "")[:10]
    end_raw = (course.get("end_date") or "")[:10]
    if start_raw and end_raw:
        return f"{start_raw} to {end_raw} ({tz_name})"
    return "Upcoming (See registration page for exact hours)"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_aol_courses_and_teachers(city: str) -> str:
    """
    Queries the official Art of Living Course Finder API (`https://unity.artofliving.org/csapi/courses`)
    used by `https://www.artofliving.org/ca-en/find/course` for the participant's city.
    Returns live upcoming courses with dates, session timings, venue, registration link,
    and teacher/volunteer contact details (name, phone, email).
    """
    geo = geocode_city(city)
    if not geo:
        city_slug = quote_plus(city)
        return (
            f"Could not geocode '{city}'. Share the official Course Finder link with the user so they can search directly: "
            f"[Find Courses in {city}](https://www.artofliving.org/ca-en/find/course#type=search&search_place={city_slug}&distance=50&limit=100&all_online=1&language=en-ca&country=ca&mctype=1)"
        )

    lat, lng, country_code, display_label = geo
    lang_code = COUNTRY_LANGUAGE_MAP.get(country_code, "en-ca")
    city_encoded = quote_plus(city.strip())

    # Direct link to the official Art of Living Course Finder pre-filled with the participant's location
    finder_page_url = (
        f"https://www.artofliving.org/{lang_code}/find/course"
        f"#type=search&lat={lat}&lng={lng}&search_place={city_encoded}"
        f"&distance=50&limit=100&all_online=1&language={lang_code}&country={country_code}&mctype=1"
    )

    try:
        resp = requests.get(
            "https://unity.artofliving.org/csapi/courses",
            params={
                "type": "search",
                "lat": lat,
                "lng": lng,
                "search_place": city.strip(),
                "distance": 50,
                "limit": 100,
                "all_online": 1,
                "language": lang_code,
                "country": country_code,
                "order_by": "start_date",
            },
            headers={"User-Agent": "Mozilla/5.0 (AOL-Ananda-Support-Assistant)"},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        raw_courses = payload.get("courses") or []
    except Exception as exc:
        return (
            f"Live API lookup for {city} encountered a temporary issue ({exc}). "
            f"Please share the direct Course Finder page for {city}: [Browse Live Courses Near {city}]({finder_page_url})"
        )

    today_str = datetime.date.today().strftime("%Y-%m-%d")

    in_person_courses = []
    online_courses = []
    teachers_contacts: dict[str, dict] = {}

    for c in raw_courses:
        start_date_str = (c.get("start_date") or "")[:10]
        if start_date_str and start_date_str < today_str:
            continue

        title = (c.get("title") or "Art of Living Course").strip()
        is_online = str(c.get("is_online_event", "0")) == "1"
        mode_label = "Online (Live Interactive)" if is_online else "In-Person"

        # Venue formatting
        if is_online:
            venue_str = "Online (Live Interactive Sessions)"
        else:
            addr_parts = [
                p.strip()
                for p in [
                    c.get("address") or c.get("address_short") or "",
                    c.get("city") or "",
                    c.get("state") or "",
                ]
                if p and p.strip()
            ]
            # Deduplicate consecutive identical address tokens
            venue_str = ", ".join(dict.fromkeys(addr_parts)) or f"{city.title()} Center"

        # Schedule & session timings
        schedule_str = _format_course_schedule(c)

        # Teachers & Volunteer/Coordinator contact details
        teacher_list = [t.strip() for t in (c.get("teachers") or []) if t and t.strip()]
        teacher_names = ", ".join(teacher_list) if teacher_list else "Art of Living Faculty"
        contact_name = (c.get("contact_name") or teacher_names).strip()
        phones_list = [p.strip() for p in (c.get("phones") or []) if p and p.strip()]
        phone_str = ", ".join(phones_list) if phones_list else "See registration link"
        email_str = (c.get("email") or "").strip()

        # Registration URL
        reg_url = (c.get("register_url") or "").strip()
        if not reg_url and c.get("link"):
            link_val = c["link"].strip()
            reg_url = link_val if link_val.startswith("http") else f"https://{link_val}"
        if not reg_url:
            reg_url = finder_page_url

        # Format contact line
        contact_parts = []
        if teacher_list:
            contact_parts.append(f"Teacher(s): **{teacher_names}**")
        if contact_name and contact_name not in teacher_names:
            contact_parts.append(f"Contact/Volunteer: **{contact_name}**")
        elif not teacher_list and contact_name:
            contact_parts.append(f"Contact/Volunteer: **{contact_name}**")
        if phones_list:
            contact_parts.append(f"Phone: `{phone_str}`")
        if email_str:
            contact_parts.append(f"Email: [{email_str}](mailto:{email_str})")
        contact_summary = " | ".join(contact_parts)

        course_entry = (
            f"- **{title}** ({mode_label})\n"
            f"  - **Dates & Timings:** {schedule_str}\n"
            f"  - **Venue:** {venue_str}\n"
            f"  - **Teacher / Volunteer Contact:** {contact_summary}\n"
            f"  - **Registration Link:** [Register Here]({reg_url})"
        )

        if is_online:
            online_courses.append(course_entry)
        else:
            in_person_courses.append(course_entry)

        # Record local teacher / volunteer contacts (prioritize in-person & contacts with phone/email)
        if not is_online or len(teachers_contacts) < 8:
            person_key = f"{contact_name}|{email_str}"
            if person_key not in teachers_contacts and (phones_list or email_str):
                teachers_contacts[person_key] = {
                    "teacher": teacher_names,
                    "contact": contact_name,
                    "city": c.get("city") or city.title(),
                    "phone": phone_str,
                    "email": email_str,
                    "program": title,
                }

    lines = [
        f"LIVE ART OF LIVING COURSES & TEACHER CONTACTS FOR {city.upper()} ({display_label}):",
        f"Official Filtered Course Finder Page for {city.title()}: [View All Courses Near {city.title()} on artofliving.org]({finder_page_url})",
        "",
    ]

    if teachers_contacts:
        lines.append(f"### Local Teachers & Course Organizers Near {city.title()}:")
        for idx, info in enumerate(list(teachers_contacts.values())[:10], start=1):
            email_md = f"[{info['email']}](mailto:{info['email']})" if info["email"] else "N/A"
            role_label = (
                f"Teacher: **{info['teacher']}** (Contact: {info['contact']})"
                if info["contact"] != info["teacher"]
                else f"**{info['teacher']}**"
            )
            lines.append(
                f"{idx}. {role_label} — {info['program']} ({info['city']})\n"
                f"   - Phone: `{info['phone']}` | Email: {email_md}"
            )
        lines.append("")

    if in_person_courses:
        lines.append(f"### Upcoming In-Person Courses in & around {city.title()} (within 50 km):")
        lines.extend(in_person_courses[:12])
        lines.append("")
    else:
        lines.append(
            f"*(No in-person courses were listed within 50 km of {city.title()} right now; see live online courses below or check the [Course Finder Page]({finder_page_url}).)*\n"
        )

    if online_courses:
        lines.append(f"### Upcoming Live Online Courses Available in {city.title()}:")
        lines.extend(online_courses[:8])
        lines.append("")

    return "\n".join(lines)


def get_local_directory_context(city: str | None) -> str:
    """
    If the participant's city is known, fetches real-time courses, dates, timings,
    venues, registration links, and teacher/volunteer phone & email contacts from
    artofliving.org. If the city is not yet known, instructs Ananda to ask the participant
    for their city/location rather than assuming Toronto.
    """
    if not city:
        return (
            "LOCAL COURSES & TEACHERS DIRECTORY STATUS:\n"
            "- The participant's city/location is NOT known yet.\n"
            "- Do NOT assume Toronto or make up course dates/venues.\n"
            "- If the participant asks about upcoming courses, dates, venues, or local teachers/volunteers (or if you are recommending a course), warmly ask them **which city they are located in** so you can look up live upcoming courses, dates, session timings, venue address, registration link, and local teacher/volunteer contact details (phone & email) near them!"
        )
    return fetch_live_aol_courses_and_teachers(city)


# ======================================================================================
# 6. STREAMLIT SESSION STATE INITIALIZATION
# ======================================================================================

# Chat messages log: [{"role": "user"|"assistant", "content": "...", "sheets": [...]}]
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User profile: remembers experience level and location across conversation turns
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = {
        "experience": None,  # "learned_sk" (practitioner) or "new" (newcomer)
        "city": None,        # Participant's city (asked rather than assumed)
    }

# ======================================================================================
# 7. WELCOME, ONBOARDING QUICK-REPLY CHIPS & PARTICIPANT LOCATION INPUT
# ======================================================================================

profile = st.session_state["user_profile"]

# Compact location bar so the participant can set or change their city at any time
with st.container():
    loc_col1, loc_col2 = st.columns([3, 2])
    with loc_col1:
        city_input = st.text_input(
            "📍 Your City / Location (for live local courses, venues & teacher contacts):",
            value=profile.get("city") or "",
            placeholder="Type your city (e.g., Vancouver, Ottawa, Chicago, Mumbai)...",
            key="city_input_box",
        )
        if city_input.strip() and city_input.strip() != (profile.get("city") or ""):
            st.session_state["user_profile"]["city"] = city_input.strip()
            profile["city"] = city_input.strip()
    with loc_col2:
        if not profile.get("city"):
            st.caption("💡 Share your city above or in chat to find live courses & teachers near you.")

# Present onboarding choice on first load before any messages are sent
if profile["experience"] is None and len(st.session_state["messages"]) == 0:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi! I'm **Ananda**, your AI Volunteer from The Art of Living. 🌿\n\n"
            "To help me tailor wisdom and share programs happening near you, have you practiced **Sudarshan Kriya** before?"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✨ Yes, I practice it!", use_container_width=True):
                st.session_state["user_profile"]["experience"] = "learned_sk"
                city_prompt = (
                    f" Since you're in **{profile['city']}**, just ask anytime for upcoming Part 2, Sahaj Samadhi, or follow-up sessions near you!"
                    if profile.get("city")
                    else " Which **city** are you located in? Feel free to share your city anytime so I can look up live local courses, follow-ups, and teacher contacts near you — or ask me any question on Gurudev's wisdom!"
                )
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Wonderful! So glad to connect with a fellow practitioner. 🌸{city_prompt}",
                    }
                )
                st.rerun()
        with col2:
            if st.button("🌱 No, I'm new!", use_container_width=True):
                st.session_state["user_profile"]["experience"] = "new"
                city_prompt = (
                    f" Whenever you're ready, I can share live upcoming beginner courses and teacher contacts right in **{profile['city']}**!"
                    if profile.get("city")
                    else " If you'd like to explore an upcoming course near you, just let me know **which city you're in**, or ask me anything about meditation, breathing techniques, and Gurudev's wisdom!"
                )
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"A warm welcome! It's so beautiful to have you here. 🌿{city_prompt}",
                    }
                )
                st.rerun()

# ======================================================================================
# 8. RENDER CHAT HISTORY
# ======================================================================================

for idx, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg.get("sheets"):
                render_knowledge_sheets_bar(msg["sheets"], idx)

# ======================================================================================
# 9. CHAT INPUT & RAG EXECUTION PIPELINE
# ======================================================================================

user_input = st.chat_input("Ask a question or share your city to find live courses near you...")

if user_input:
    # 1. Append user message to history and display it immediately
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 2. Extract / update user's city if mentioned in the query
    detected_city = detect_and_update_city(user_input, profile.get("city"))
    if detected_city:
        st.session_state["user_profile"]["city"] = detected_city
        profile["city"] = detected_city

    # 3. Build personalized context guidance for Claude
    exp = profile.get("experience")
    city = profile.get("city")
    user_context_notes = []
    if exp == "learned_sk":
        user_context_notes.append(
            "User has already completed the Art of Living Part 1 course and practices Sudarshan Kriya. "
            "When relevant, gently and warmly (without being pushy) encourage them to deepen their journey with the Art of Living Part 2 course (Silence Program) or Sahaj Samadhi Meditation, and remind them of regular practice or local follow-ups."
        )
    elif exp == "new":
        user_context_notes.append(
            "User is new to The Art of Living and has NOT done any course yet. "
            "When relevant, gently and warmly (never pushy) recommend the Art of Living Part 1 course (Sudarshan Kriya) or Sahaj Samadhi Meditation as the ultimate gold-standard foundational practices to begin with."
        )
    else:
        user_context_notes.append(
            "If the user is new, gently suggest the Art of Living Part 1 course or Sahaj Samadhi Meditation as the gold-standard starting point; if they already completed Part 1, gently mention Part 2 as a beautiful next step (never be pushy)."
        )
    if city:
        user_context_notes.append(f"Participant is located in or looking for courses near: {city}.")
    else:
        user_context_notes.append(
            "Participant's city is NOT known yet. If they ask for courses, schedules, venues, or local teachers/volunteers without specifying a city, warmly ask them which city they live in so you can fetch live courses and teacher contacts for their exact location."
        )

    # 4. Retrieve LIVE courses & teacher/volunteer contacts from artofliving.org for the participant's city
    local_directory_info = get_local_directory_context(city)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # ----------------------------------------------------------------------
                # Step A: AWS Bedrock Knowledge Base Retrieval (Vector Search)
                # ----------------------------------------------------------------------
                retrieval_response = bedrock_agent_client.retrieve(
                    knowledgeBaseId=KNOWLEDGE_BASE_ID,
                    retrievalQuery={"text": user_input},
                )
                retrieval_results = retrieval_response.get("retrievalResults", [])

                # Filter results by relevance score (threshold >= 0.35)
                relevant_results = [
                    r for r in retrieval_results if r.get("score", 1.0) >= 0.35
                ]
                if not relevant_results:
                    relevant_results = retrieval_results

                # ----------------------------------------------------------------------
                # Step B: Build Indexed Knowledge Sheets & Context Excerpts
                # ----------------------------------------------------------------------
                sheets = build_knowledge_sheets(relevant_results) if relevant_results else []

                context_blocks = []
                for s in sheets:
                    chunks_text = "\n".join(s["chunks"])
                    context_blocks.append(
                        f"[{s['index']}] Knowledge Sheet: {s['title']} (URL: {s['url']})\n{chunks_text}"
                    )

                joined_context = "\n\n".join(context_blocks)

                # ----------------------------------------------------------------------
                # Step C: Formulate the System Prompt (Persona, Citations & Live Courses)
                # ----------------------------------------------------------------------
                system_prompt = (
                    "You are 'Ananda', a warm, compassionate, and joyful AI volunteer for The Art of Living.\n\n"
                    "Persona & Tone Guidelines:\n"
                    "- Speak with warmth, compassion, calmness, and joy — just like a seasoned Art of Living volunteer.\n"
                    "- Greeting: Do NOT use 'Namaste'. Use a friendly greeting of their country if known, otherwise simply say 'Hi' or 'Hello'.\n"
                    "- Refer to Sri Sri Ravi Shankar respectfully as 'Gurudev', informally and naturally — do NOT say 'Gurudev says' or 'Gurudev teaches' after every line. Mention 'Gurudev' once (or at most twice, or thrice only if truly needed in a long answer) and let the rest of the wisdom flow naturally.\n"
                    "- Answer clearly, upliftingly, and accurately using the provided Knowledge Sheet excerpts.\n"
                    "- CITATION INSTRUCTION: Whenever you share a perspective, quote, or teaching from an excerpt, append its reference tag like [1], [2], etc., right after that sentence or thought. This lets users explore that exact teaching.\n"
                    "- Course Recommendations (Helpful & Never Pushy):\n"
                    "  * For someone who hasn't done any course yet: warmly recommend the **Art of Living Part 1 Course** (Sudarshan Kriya) or **Sahaj Samadhi Meditation** ([Explore Courses](https://www.artofliving.org)) as the ultimate gold-standard practices to begin their journey.\n"
                    "  * For someone who has already done Part 1 / practices Sudarshan Kriya: gently encourage exploring the **Art of Living Part 2 Course** (Silence Program) or **Sahaj Samadhi Meditation** to go deeper, without ever sounding pushy.\n"
                    "- Live Local Courses & Teacher/Volunteer Contacts (Fetched directly from artofliving.org):\n"
                    "  * NEVER invent or fabricate course dates, venues, or teacher names/phone numbers. Use ONLY the live courses and teacher/volunteer contacts provided in the LIVE ART OF LIVING COURSES section below.\n"
                    "  * If the participant's city is NOT known yet and they ask about courses, dates, or local teachers, warmly ask them which city they are located in so you can pull up the exact upcoming courses, venues, registration links, and teacher/volunteer contact details for their city.\n"
                    "  * When the participant's city IS known and they share their city or ask about courses/teachers, present the live courses near their city with:\n"
                    "    1. **Course Name & Format** (In-Person or Online, prioritizing In-Person near their city)\n"
                    "    2. **Exact Dates & Day-by-Day Session Timings**\n"
                    "    3. **Full Venue Address**\n"
                    "    4. **Teacher / Volunteer Name, Phone Number, and Email Address**\n"
                    "    5. **Direct Clickable Registration Link** (plus the link to view all courses near their city on `artofliving.org`)\n"
                    "- Meditation Resources (When the user asks about meditation, how to meditate, or calming the mind):\n"
                    "  * Share Gurudev's online guided meditations on YouTube: [Gurudev's Guided Meditations on YouTube](https://www.youtube.com/@MeditationsByGurudev)\n"
                    "  * Share the **Sattva Meditation App** links for both iOS and Android:\n"
                    "    - [Sattva App on iOS App Store](https://apps.apple.com/app/id1111585505)\n"
                    "    - [Sattva App on Google Play (Android)](https://play.google.com/store/apps/details?id=com.sattva.life)\n"
                    "  * Also recommend **Art of Living Part 1** and **Sahaj Samadhi Meditation** as the ultimate gold-standard practices to truly begin and establish a meditation journey.\n"
                    "- If the excerpts do not contain enough information, acknowledge that gently and offer practical wisdom or invite them to a local Art of Living center or follow-up.\n"
                    "- Always maintain an encouraging, grounded, and peaceful presence.\n"
                    "- Answer naturally without technical disclaimers."
                )

                if user_context_notes:
                    system_prompt += f"\n\nUser Background:\n{' '.join(user_context_notes)}"

                system_prompt += f"\n\n{local_directory_info}"

                if joined_context:
                    user_prompt = (
                        f"Knowledge Sheet Excerpts:\n{joined_context}\n\n"
                        f"User Question: {user_input}\n\n"
                        "Please answer the user's question, citing the relevant excerpts with [1], [2], etc."
                    )
                else:
                    user_prompt = user_input

                # ----------------------------------------------------------------------
                # Step D: Invoke Claude via AWS Bedrock Runtime
                # ----------------------------------------------------------------------
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1200,
                    "temperature": 0.7,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                    ],
                }

                model_response = bedrock_client.invoke_model(
                    modelId=MODEL_ID,
                    accept=accept,
                    contentType=contentType,
                    body=json.dumps(payload),
                )
                model_output = json.loads(model_response.get("body").read())
                raw_answer = model_output["content"][0]["text"].strip()

                # ----------------------------------------------------------------------
                # Step E: Post-process & Render Response with Interactive Citations
                # ----------------------------------------------------------------------
                # Injects custom HTML hover badges into [1], [2] tags
                rendered_answer = inject_hover_citations(raw_answer, sheets)
                final_reply = rendered_answer + "\n\n_— Ananda, AI Volunteer_"

                st.markdown(final_reply, unsafe_allow_html=True)

                # Render referenced sheets bar at the bottom with expand buttons
                new_msg_idx = len(st.session_state["messages"])
                if sheets:
                    render_knowledge_sheets_bar(sheets, new_msg_idx)

                # Persist to session history
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": final_reply,
                        "sheets": sheets,
                    }
                )

            except Exception as e:
                err_msg = f"⚠️ An error occurred: {str(e)}"
                st.markdown(err_msg)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": err_msg}
                )
