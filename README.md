# AOL Wisdom Support — AI Volunteer "Ananda"

An intelligent, compassionate conversational AI assistant and Retrieval-Augmented Generation (RAG) system built for **The Art of Living**. 

Powered by **Amazon Bedrock Knowledge Bases**, **Anthropic Claude 3.5**, and **Streamlit**, this application provides personalized wisdom guidance, meditation resources, interactive Knowledge Sheet citation reading, and local community connections (teachers & upcoming course schedules).

---

## 🌟 Key Features

1. **Decoupled RAG Pipeline:**
   - **Vector Search:** Queries an authoritative AWS Bedrock Knowledge Base (crawled wisdom articles and discourses from Gurudev Sri Sri Ravi Shankar).
   - **Synthesis:** Powered by Claude on Amazon Bedrock Runtime to deliver warm, empathetic, and grounded answers.

2. **Interactive Inline Citations (`[1]`, `[2]`):**
   - Whenever an answer draws upon a specific teaching, an interactive citation badge is inserted directly into the text.
   - **Hover Preview:** Hovering over any citation badge opens a floating, scrollable, and resizable window with the exact excerpt from the source article.
   - **Full-Screen Reader:** Click **⤢ Expand Full Screen** to read the complete article word-by-word in a clean modal dialog.

3. **Persona & Tone of "Ananda" (AI Volunteer):**
   - Communicates with genuine warmth, calmness, and joy — just like a seasoned Art of Living volunteer.
   - Respectfully refers to Sri Sri Ravi Shankar as *"Gurudev"*, weaving teachings naturally into the conversation without repetitive phrasing.

4. **Personalized Journey & Onboarding Quick-Chips:**
   - On first arrival, users can choose their background with a single tap:
     - `✨ Yes, I practice it!` (Experienced practitioner of Sudarshan Kriya)
     - `🌱 No, I'm new!` (New to The Art of Living)
   - Tailors wisdom and guidance accordingly:
     - **For beginners:** Recommends the **Art of Living Part 1 Course** (Sudarshan Kriya) or **Sahaj Samadhi Meditation** as gold-standard starting points.
     - **For practitioners:** Gently encourages deepening the journey with the **Art of Living Part 2 Course** (Silence Retreat) and regular practice.

5. **Meditation & App Resources:**
   - When asked about meditation, provides direct links to:
     - [Gurudev's Guided Meditations on YouTube](https://www.youtube.com/@MeditationsByGurudev)
     - **Sattva Meditation App** for [iOS App Store](https://apps.apple.com/app/id1111585505) and [Google Play (Android)](https://play.google.com/store/apps/details?id=com.sattva.life)

6. **Local Teachers & Upcoming 2–4 Week Schedules:**
   - Automatically detects user location (e.g. *Toronto*, *Mississauga*, *Vancouver*, *New York*, etc.).
   - Shares a complete directory of local teachers with **Names, Focus Areas, Phone Numbers, and Emails**.
   - Generates upcoming course schedules in the coming 2 to 4 weeks with **Dates, Detailed Timings, Venues, and Clickable Registration Links**.

---

## 🏗️ Architecture

```
User Query (e.g., "I'm feeling heartbroken, what did Gurudev say?")
                         │
                         ▼
             [Location & Context Detection]
                         │
                         ▼
        [AWS Bedrock Knowledge Base Retrieval]
           (Vector Search across wisdom docs)
                         │
                         ▼
           [System Prompt + Context Assembly]
           (Ananda Persona + Wisdom Chunks +
            Local Teachers & Upcoming Schedule)
                         │
                         ▼
        [Anthropic Claude on AWS Bedrock Runtime]
                         │
                         ▼
     [Interactive Citation & Hover Post-Processor]
                         │
                         ▼
         Streamlit UI with Floating Popovers,
         Full-Screen Dialogs & Local Directory
```

---

## 🚀 Setup & Installation Guide

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- An active **AWS Account** with:
  - Permissions for **Amazon Bedrock** (`bedrock:Retrieve`, `bedrock:InvokeModel`).
  - Model access enabled for **Anthropic Claude 3.5 Sonnet / Claude Opus** in your AWS Bedrock console.
  - A configured **AWS Bedrock Knowledge Base** with your indexed content.

---

### 2. Clone the Repository
```bash
git clone https://github.com/dipps511/aol-wisdom-support.git
cd aol-wisdom-support
```

---

### 3. Create and Activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate
```

---

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 5. Configure AWS Credentials
Make sure your environment has valid AWS credentials configured. You can use the AWS CLI:

```bash
aws configure
```
Provide:
- **AWS Access Key ID**
- **AWS Secret Access Key**
- **Default region name** (e.g., `us-east-2`)
- **Default output format** (`json`)

Alternatively, export environment variables:
```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-2"
```

---

### 6. Update Configuration in `main.py`
Open `main.py` and adjust the configuration parameters near lines 30–45 if yours differ:

```python
# --- CONFIGURATION ---
REGION = "us-east-2"
MODEL_ID = "us.anthropic.claude-opus-4-5-20251101-v1:0" # Or your Claude model ARN/ID
KNOWLEDGE_BASE_ID = "YOUR_KNOWLEDGE_BASE_ID"           # Replace with your Bedrock KB ID
```

---

### 7. Run the Application
Launch the Streamlit app:

```bash
streamlit run main.py
```
Or via Python module:
```bash
python -m streamlit run main.py
```

Open your browser at **`http://localhost:8502`** (or `http://localhost:8501`).

---

## 📁 Project Structure

```
.
├── main.py              # Main application logic: RAG pipeline, UI, citations, and directory
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation and setup instructions
```

---

## 🛠️ Tech Stack

- **Frontend / Application Framework:** [Streamlit](https://streamlit.io/)
- **LLM & Inference:** [Anthropic Claude 3.5](https://www.anthropic.com/) via [Amazon Bedrock](https://aws.amazon.com/bedrock/)
- **Vector Search / Knowledge Base:** [Amazon Bedrock Knowledge Bases](https://aws.amazon.com/bedrock/knowledge-bases/)
- **AWS SDK:** [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- **Web Scraping & HTML Sanitization:** [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) & [Requests](https://requests.readthedocs.io/)

---

## 📜 License
This project is open-source and intended for educational, community, and support purposes. Teachings and wisdom are attributed to **The Art of Living Foundation** and **Sri Sri Ravi Shankar**.
