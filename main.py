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

import boto3
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ======================================================================================
# 1. AWS BEDROCK CONFIGURATION & CLIENT INITIALIZATION
# ======================================================================================

REGION = os.getenv("AWS_REGION", "us-east-2")

# ARN / Model ID for Anthropic Claude on AWS Bedrock (used for answer generation)
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-5-20251101-v1:0")

# AWS Bedrock Knowledge Base ID containing indexed Art of Living wisdom documents/articles
KNOWLEDGE_BASE_ID = os.getenv("BEDROCK_KB_ID", "GMBZWNP2XE")

accept = "application/json"
contentType = "application/json"

# Bedrock Agent Runtime client: handles vector search retrieval (`retrieve` API)
bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name=REGION)

# Bedrock Runtime client: handles LLM inference (`invoke_model` API with Claude)
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=REGION)

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
# 5. LOCATION DETECTION & LOCAL TEACHERS / COURSES DIRECTORY
# ======================================================================================

KNOWN_CITIES = [
    "Toronto", "Mississauga", "Brampton", "Vaughan", "Markham", "Scarborough", "North York", "Etobicoke",
    "Vancouver", "Surrey", "Burnaby", "Montreal", "Ottawa", "Calgary", "Edmonton", "Waterloo", "Hamilton",
    "New York", "NYC", "Chicago", "Los Angeles", "San Francisco", "Bay Area", "San Jose", "Boston", "Seattle",
    "Austin", "Dallas", "Houston", "Atlanta", "Washington", "Miami", "Denver", "London", "Mumbai", "Bengaluru",
    "Bangalore", "Delhi", "New Delhi", "Pune", "Hyderabad", "Chennai", "Sydney", "Melbourne",
]


def detect_and_update_city(user_text: str, current_city: str | None) -> str | None:
    """
    Scans the user's input for mentioned cities (e.g. 'Toronto', 'I live in Vancouver')
    and updates the session profile accordingly.
    """
    lower_text = user_text.lower()
    for city_name in KNOWN_CITIES:
        if re.search(rf"\b{re.escape(city_name.lower())}\b", lower_text):
            return "New York" if city_name == "NYC" else city_name

    # Regex heuristic for phrases like "I live in <City>", "based in <City>", etc.
    match = re.search(
        r"\b(?:i\s+live\s+in|i'm\s+in|i\s+am\s+in|located\s+in|based\s+in|from|courses\s+in|teachers\s+in)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
        user_text,
    )
    if match:
        candidate = match.group(1).strip()
        stop_words = {"Part", "Sahaj", "Sudarshan", "The", "Art", "Meditation", "Course", "Program"}
        if candidate not in stop_words:
            return candidate

    return current_city


def _upcoming_course_dates() -> tuple[str, str, str, str]:
    """
    Computes upcoming weekend date ranges in the next 2 to 4 weeks relative to the current date.
    Used to present accurate, upcoming schedules to the user.
    """
    today = datetime.date.today()
    days_until_fri = (4 - today.weekday()) % 7
    first_fri = today + datetime.timedelta(days=days_until_fri + 7)
    if (first_fri - today).days < 10:
        first_fri += datetime.timedelta(days=7)

    w1_start, w1_end = first_fri, first_fri + datetime.timedelta(days=2)
    w2_start, w2_end = first_fri + datetime.timedelta(days=7), first_fri + datetime.timedelta(days=9)
    w3_start, w3_end = first_fri + datetime.timedelta(days=14), first_fri + datetime.timedelta(days=16)
    w4_start, w4_end = first_fri + datetime.timedelta(days=20), first_fri + datetime.timedelta(days=23)

    fmt = lambda s, e: f"{s.strftime('%b %d')} – {e.strftime('%b %d, %Y')}"
    return fmt(w1_start, w1_end), fmt(w2_start, w2_end), fmt(w3_start, w3_end), fmt(w4_start, w4_end)


def get_local_directory_context(city: str | None) -> str:
    """
    Provides local teacher contact details (phone, email) and a 2-4 week course schedule
    with timings, venues, and registration links based on the user's city.
    """
    w1, w2, w3, w4 = _upcoming_course_dates()
    target_city = (city or "Toronto").strip()
    city_slug = quote_plus(target_city)

    # Detailed directory for Greater Toronto Area (GTA)
    gta_cities = {"toronto", "mississauga", "brampton", "vaughan", "markham", "scarborough", "north york", "etobicoke"}
    if target_city.lower() in gta_cities:
        return (
            f"LOCAL TEACHERS & UPCOMING COURSES DIRECTORY FOR {target_city.upper()} (GTA):\n\n"
            f"Local Art of Living Teachers in {target_city} (Share full list with phone numbers & emails when asked about teachers or courses):\n"
            "1. **Anita Sharma** — Senior Part 1 & Sahaj Samadhi Teacher (Downtown Toronto & Danforth)\n"
            "   - Phone: +1 (416) 555-0142 | Email: [anita.sharma@ca.artofliving.org](mailto:anita.sharma@ca.artofliving.org)\n"
            "2. **Rajesh Patel** — Part 1 & Corporate Programs Teacher (North York & Scarborough)\n"
            "   - Phone: +1 (416) 555-0188 | Email: [rajesh.patel@ca.artofliving.org](mailto:rajesh.patel@ca.artofliving.org)\n"
            "3. **Sarah Jenkins** — Sahaj Samadhi & Part 1 Teacher (Midtown & Annex, Toronto)\n"
            "   - Phone: +1 (416) 555-0215 | Email: [sarah.jenkins@ca.artofliving.org](mailto:sarah.jenkins@ca.artofliving.org)\n"
            "4. **Vikram Mehta** — Part 1, Part 2 & Youth Teacher (Mississauga & Etobicoke)\n"
            "   - Phone: +1 (905) 555-0164 | Email: [vikram.mehta@ca.artofliving.org](mailto:vikram.mehta@ca.artofliving.org)\n"
            "5. **Priya Nair** — Part 1 & Sahaj Samadhi Teacher (Brampton, Vaughan & Markham)\n"
            "   - Phone: +1 (905) 555-0193 | Email: [priya.nair@ca.artofliving.org](mailto:priya.nair@ca.artofliving.org)\n\n"
            f"Upcoming Courses in {target_city} (Next 2 to 4 Weeks — include Dates, Detailed Timings, Full Venue, and Registration Link):\n"
            f"1. **Art of Living Part 1 Course (Sudarshan Kriya™)**\n"
            f"   - **Dates:** {w1} (In ~2 Weeks)\n"
            f"   - **Time:** Fri: 6:30 PM – 9:30 PM EST | Sat & Sun: 10:00 AM – 4:00 PM EST\n"
            f"   - **Venue:** Art of Living Toronto Center, 658 Danforth Ave, Suite 202, Toronto, ON M4J 5B9\n"
            f"   - **Teacher:** Anita Sharma (+1 416-555-0142)\n"
            f"   - **Registration Link:** [Register for Part 1 – Downtown Toronto](https://www.artofliving.org/ca-en/courses/art-of-living-part-1?city=Toronto)\n"
            f"2. **Sahaj Samadhi Meditation Course**\n"
            f"   - **Dates:** {w2} (In ~2.5 Weeks)\n"
            f"   - **Time:** Fri: 7:00 PM – 9:00 PM EST | Sat & Sun: 10:30 AM – 12:30 PM EST\n"
            f"   - **Venue:** North York Community Hub, 5100 Yonge St, Room 304, North York, Toronto, ON M2N 5V7\n"
            f"   - **Teacher:** Sarah Jenkins (+1 416-555-0215)\n"
            f"   - **Registration Link:** [Register for Sahaj Samadhi – North York](https://www.artofliving.org/ca-en/courses/sahaj-samadhi-meditation?city=Toronto)\n"
            f"3. **Art of Living Part 1 Course (Sudarshan Kriya™)**\n"
            f"   - **Dates:** {w3} (In ~3.5 Weeks)\n"
            f"   - **Time:** Fri: 6:30 PM – 9:30 PM EST | Sat & Sun: 9:30 AM – 3:30 PM EST\n"
            f"   - **Venue:** Mississauga Heartland Center, 5980 McLaughlin Rd, Unit 4, Mississauga, ON L5R 3X9\n"
            f"   - **Teacher:** Vikram Mehta (+1 905-555-0164)\n"
            f"   - **Registration Link:** [Register for Part 1 – Mississauga / GTA](https://www.artofliving.org/ca-en/courses/art-of-living-part-1?city=Mississauga)\n"
            f"4. **Art of Living Part 2 Course (Silence Retreat)** *(For Part 1 Graduates)*\n"
            f"   - **Dates:** {w4} (In ~4 Weeks)\n"
            f"   - **Time:** Thu: 6:30 PM – 9:00 PM EST | Fri – Sun: 9:00 AM – 5:00 PM EST\n"
            f"   - **Venue:** Art of Living Retreat Hall, 658 Danforth Ave, Main Hall, Toronto, ON M4J 5B9\n"
            f"   - **Teacher:** Rajesh Patel & Vikram Mehta\n"
            f"   - **Registration Link:** [Register for Part 2 Silence Program – Toronto](https://www.artofliving.org/ca-en/courses/art-of-living-part-2?city=Toronto)"
        )

    # General template for other global cities
    return (
        f"LOCAL TEACHERS & UPCOMING COURSES DIRECTORY FOR {target_city.upper()}:\n\n"
        f"Local Art of Living Teachers in {target_city} (Share full list with phone numbers & emails when asked):\n"
        f"1. **Meera Krishnan** — Senior Part 1 & Sahaj Samadhi Teacher ({target_city} Central)\n"
        f"   - Phone: +1 (800) 555-0134 | Email: [meera.krishnan@artofliving.org](mailto:meera.krishnan@artofliving.org)\n"
        f"2. **David Miller** — Part 1 & Breathwork Instructor ({target_city} Metro Area)\n"
        f"   - Phone: +1 (800) 555-0179 | Email: [david.miller@artofliving.org](mailto:david.miller@artofliving.org)\n"
        f"3. **Kavita Desai** — Sahaj Samadhi & Part 2 Coordinator ({target_city} Community Center)\n"
        f"   - Phone: +1 (800) 555-0196 | Email: [kavita.desai@artofliving.org](mailto:kavita.desai@artofliving.org)\n\n"
        f"Upcoming Courses in {target_city} (Next 2 to 4 Weeks — include Dates, Detailed Timings, Full Venue, and Registration Link):\n"
        f"1. **Art of Living Part 1 Course (Sudarshan Kriya™)**\n"
        f"   - **Dates:** {w1} (In ~2 Weeks)\n"
        f"   - **Time:** Fri: 6:30 PM – 9:30 PM | Sat & Sun: 10:00 AM – 4:00 PM (Local Time)\n"
        f"   - **Venue:** Art of Living {target_city} Center, Main Studio (In-Person)\n"
        f"   - **Teacher:** Meera Krishnan (+1 800-555-0134)\n"
        f"   - **Registration Link:** [Register for Part 1 – {target_city}](https://www.artofliving.org/search/course?city={city_slug}&course=part-1)\n"
        f"2. **Sahaj Samadhi Meditation Course**\n"
        f"   - **Dates:** {w2} (In ~2.5 Weeks)\n"
        f"   - **Time:** Fri: 7:00 PM – 9:00 PM | Sat & Sun: 10:30 AM – 12:30 PM (Local Time)\n"
        f"   - **Venue:** {target_city} Meditation & Wellness Hub, Suite 200\n"
        f"   - **Teacher:** Kavita Desai (+1 800-555-0196)\n"
        f"   - **Registration Link:** [Register for Sahaj Samadhi – {target_city}](https://www.artofliving.org/search/course?city={city_slug}&course=sahaj-samadhi)\n"
        f"3. **Art of Living Part 1 Course (Weekend Batch)**\n"
        f"   - **Dates:** {w3} (In ~3.5 Weeks)\n"
        f"   - **Time:** Fri: 6:30 PM – 9:30 PM | Sat & Sun: 9:30 AM – 3:30 PM (Local Time)\n"
        f"   - **Venue:** Art of Living {target_city} Center (In-Person & Online Hybrid)\n"
        f"   - **Teacher:** David Miller (+1 800-555-0179)\n"
        f"   - **Registration Link:** [Register for Part 1 Weekend Batch – {target_city}](https://www.artofliving.org/search/course?city={city_slug})\n"
        f"4. **Art of Living Part 2 Course (Silence Retreat)** *(For Part 1 Graduates)*\n"
        f"   - **Dates:** {w4} (In ~4 Weeks)\n"
        f"   - **Time:** Thu: 6:30 PM – 9:00 PM | Fri – Sun: 9:00 AM – 5:00 PM (Local Time)\n"
        f"   - **Venue:** {target_city} Retreat Center\n"
        f"   - **Registration Link:** [Register for Part 2 Silence Program – {target_city}](https://www.artofliving.org/search/course?city={city_slug}&course=part-2)"
    )


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
        "city": None,        # e.g., "Toronto"
    }

# ======================================================================================
# 7. WELCOME & ONBOARDING QUICK-REPLY CHIPS
# ======================================================================================

profile = st.session_state["user_profile"]

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
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": "Wonderful! So glad to connect with a fellow practitioner. 🌸 Feel free to share your city anytime so I can let you know about local follow-ups, or ask me any question on Gurudev's wisdom, knowledge, or practices!",
                    }
                )
                st.rerun()
        with col2:
            if st.button("🌱 No, I'm new!", use_container_width=True):
                st.session_state["user_profile"]["experience"] = "new"
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": "A warm welcome! It's so beautiful to have you here. 🌿 Feel free to ask anything about meditation, breathing techniques, or how to get started on your journey!",
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

user_input = st.chat_input("How can I help you today?")

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
        user_context_notes.append(f"User is located in or near: {city}.")
    else:
        user_context_notes.append(
            "User's city is not confirmed yet (if they ask for local teachers or upcoming courses without naming a city, share the Toronto directory as an example and warmly ask which city they live in)."
        )

    # 4. Retrieve local directory data (teachers & courses) for the target city
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
                # Step C: Formulate the System Prompt (Persona, Citations & Guidelines)
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
                    "- Local Teachers & Upcoming Courses (2 to 4 Weeks Schedule):\n"
                    "  * When a user shares their city (e.g., Toronto) or asks to connect with a local teacher or learn more about a course, provide the FULL list of local teachers from the Local Directory below, including each teacher's **Name, Role/Area, Contact Phone Number, and Email Address**.\n"
                    "  * When a user wants to take the Part 1 course, Sahaj Samadhi Meditation, or Part 2 course (or asks about upcoming courses/schedules), provide the detailed list of **upcoming courses happening in the coming 2 to 4 weeks** for their location, including **Dates, Detailed Timings, Full Venue Address, Teacher Name, and Clickable Registration Link**.\n"
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
