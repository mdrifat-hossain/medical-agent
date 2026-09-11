# Multi-Tool Medical AI Agent

A multi-tool AI agent that answers questions about three medical datasets
(Heart Disease, Cancer, Diabetes) via SQL, and general medical questions
(definitions/symptoms/cures) via web search — routed automatically to the
right tool.

> **Note on tooling:** The assignment references the OpenAI Agents SDK. This
> implementation uses **LangChain's tool-calling Agent + AgentExecutor**
> (LangChain's own agent framework, functionally equivalent to the OpenAI
> Agents SDK's tool-routing agent) powered by a **local Ollama model**
> instead of the paid OpenAI API, so the whole project runs for free/offline.
> The architecture — tools, routing logic, SQL generation, NL responses —
> matches every functional requirement in the assignment.

---

## 🏗️ Architecture

```
User question
      │
      ▼
Main Agent (LangChain AgentExecutor, local Ollama LLM)
      │
      ├── stats / numbers / data about heart disease → HeartDiseaseDBTool ──► heart_disease.db
      ├── stats / numbers / data about cancer         → CancerDBTool ───────► cancer.db
      ├── stats / numbers / data about diabetes        → DiabetesDBTool ─────► diabetes.db
      └── definitions / symptoms / cures / general Qs  → MedicalWebSearchTool ► DuckDuckGo
```

Each DB tool works in two LLM steps:
1. **NL → SQL**: the LLM reads the DB schema and writes a `SELECT` query for the question.
2. **Rows → NL**: the LLM turns the returned rows into a plain-English answer.

Only `SELECT` statements are allowed (enforced in code) — the agent can never modify the databases.

---

## 📁 Project Structure

```
medical-agent/
├── agent.py                  # Main agent entry point (routing logic lives here)
├── build_databases.py        # CSV -> SQLite conversion
├── generate_sample_data.py   # Optional: synthetic demo data (for testing without Kaggle)
├── requirements.txt
├── .env.example
├── data/                     # Put the 3 Kaggle CSVs here
├── db/                       # Generated SQLite DBs land here
└── tools/
    ├── db_tool.py             # Generic DB tool factory (SQL gen + execution + summary)
    └── web_search_tool.py     # MedicalWebSearchTool (DuckDuckGo)
```

---

## ⚙️ Setup

### 1. Install Ollama and pull a tool-calling-capable model

You already have Ollama installed. Make sure you have a model that supports
tool/function calling. From your `ollama list`, good choices are:

```
qwen3:8b            (default, recommended balance of speed/quality)
qwen2.5-coder:14b    (stronger at SQL generation, slower)
qwen2.5-coder:7b
```

Ollama must be running locally (it usually runs as a background service —
`ollama serve` if not already running).

### 2. Clone the repo and create a virtual environment

```bash
git clone <your-repo-url>
cd medical-agent

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Configure the model

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
```

Edit `.env` if you want a different model than `qwen3:8b`.

### 4. Get the datasets

Download the three CSVs from Kaggle and place them in `data/` with these exact names:

| Kaggle dataset | Save as |
|---|---|
| [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) | `data/heart.csv` |
| [Cancer Prediction Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) | `data/cancer.csv` |
| [Diabetes Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) | `data/diabetes.csv` |

**No Kaggle account handy / just want to test the pipeline first?**
Run this to generate small synthetic CSVs with the same schemas:

```bash
python generate_sample_data.py
```

### 5. Build the SQLite databases

```bash
python build_databases.py
```

This creates `db/heart_disease.db`, `db/cancer.db`, `db/diabetes.db` with
correctly typed columns inferred from each CSV.

### 6. Run the agent

```bash
python agent.py
```

---

## 💬 Example Queries

**Dataset / statistics questions (routed to DB tools):**
```
What is the average cholesterol level in the heart disease dataset?
How many patients in the diabetes dataset have a positive diabetes outcome?
What's the average BMI of patients diagnosed with cancer?
What percentage of patients in the heart disease dataset are male?
```

**General medical knowledge questions (routed to web search):**
```
What is diabetes?
What are the symptoms of lung cancer?
How is heart disease treated?
What causes high blood pressure?
```

**Mixed (agent can call multiple tools):**
```
What's the average HbA1c level in the diabetes dataset, and what does a high HbA1c level mean?
```

---

## 🔧 Troubleshooting

- **`ConnectionError` to Ollama** — make sure Ollama is running (`ollama serve`)
  and the model in `.env` has been pulled (`ollama pull qwen3:8b`).
- **Agent picks the wrong tool** — try a stronger tool-calling model like
  `qwen2.5-coder:14b` in `.env`; smaller models occasionally misroute.
- **SQL errors from a DB tool** — the tool only allows `SELECT` queries and
  will show the generated SQL error message; rephrase the question if the
  generated SQL is malformed.
- **DuckDuckGo search rate-limited** — swap to Tavily's free tier by
  uncommenting the Tavily block in `tools/web_search_tool.py` and adding
  `TAVILY_API_KEY` to `.env`.

---

## 📓 Google Colab

To run in Colab instead of locally, note that Colab can't reach your local
Ollama instance directly — either:
- run `ollama serve` via a Colab terminal add-on / ngrok tunnel, or
- swap `ChatOllama` for a hosted free-tier model (e.g. Groq, Gemini free
  tier) in `agent.py` and `tools/db_tool.py`.

---

## ✅ Requirements Checklist

- [x] CSV → SQLite conversion with typed columns (`build_databases.py`)
- [x] `heart_disease.db`, `cancer.db`, `diabetes.db` with meaningful table names
- [x] `HeartDiseaseDBTool`, `CancerDBTool`, `DiabetesDBTool` — NL question → SQL → NL answer
- [x] `MedicalWebSearchTool` — free web search for general medical knowledge
- [x] Main agent with automatic routing (stats → DB tool, knowledge → web tool)
- [x] README with run instructions
