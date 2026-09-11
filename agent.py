"""
agent.py

Main Multi-Tool Medical AI Agent.

Routes each user question to the correct tool:
  - HeartDiseaseDBTool / CancerDBTool / DiabetesDBTool for stats/numbers/data
  - MedicalWebSearchTool for definitions/symptoms/cures/general knowledge

Powered by a local Ollama model via langchain-ollama, using LangChain's
tool-calling Agent + AgentExecutor (the LangChain equivalent of the
OpenAI Agents SDK, adapted here to run fully offline/free with Ollama).

Run:
    python agent.py
"""

import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.db_tool import make_db_tool
from tools.web_search_tool import make_web_search_tool

load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:8b")

HEART_DB = "db/heart_disease.db"
CANCER_DB = "db/cancer.db"
DIABETES_DB = "db/diabetes.db"

SYSTEM_PROMPT = """You are a Multi-Tool Medical AI Agent.

You have access to four tools:
1. HeartDiseaseDBTool  - statistics/numbers from the heart disease dataset
2. CancerDBTool        - statistics/numbers from the cancer prediction dataset
3. DiabetesDBTool      - statistics/numbers from the diabetes dataset
4. MedicalWebSearchTool - general medical knowledge (definitions, symptoms, cures)

ROUTING RULES (follow strictly):
- If the question asks about statistics, counts, averages, correlations,
  distributions, or any numeric fact that would come from a dataset
  (heart disease, cancer, or diabetes patient data), use the matching
  DB tool.
- If the question asks "what is X", "what causes X", "what are the symptoms
  of X", "how is X treated/cured" — i.e. general medical knowledge not tied
  to the specific patient datasets — use MedicalWebSearchTool.
- Pick exactly one tool per question unless the question genuinely needs
  both (e.g. "What is the average cholesterol in the dataset, and what does
  high cholesterol mean?" -> use HeartDiseaseDBTool then MedicalWebSearchTool).
- If it's unclear which dataset a data question refers to, ask the user to
  clarify instead of guessing.
- Always give a direct, complete final answer in plain English.
"""


def build_agent() -> AgentExecutor:
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    heart_tool = make_db_tool(
        db_path=HEART_DB,
        tool_name="HeartDiseaseDBTool",
        description=(
            "Use for statistical or numeric questions about the heart disease "
            "patient dataset — e.g. counts, averages, distributions, or "
            "correlations involving age, cholesterol, resting blood pressure, "
            "chest pain type, max heart rate, or presence of heart disease "
            "(target). Input: a plain English question."
        ),
        model_name=MODEL_NAME,
    )

    cancer_tool = make_db_tool(
        db_path=CANCER_DB,
        tool_name="CancerDBTool",
        description=(
            "Use for statistical or numeric questions about the cancer "
            "prediction dataset — e.g. counts, averages of age/BMI, smoking "
            "rates, genetic risk, physical activity, alcohol intake, or "
            "diagnosis distribution. Input: a plain English question."
        ),
        model_name=MODEL_NAME,
    )

    diabetes_tool = make_db_tool(
        db_path=DIABETES_DB,
        tool_name="DiabetesDBTool",
        description=(
            "Use for statistical or numeric questions about the diabetes "
            "prediction dataset — e.g. HbA1c level, blood glucose level, BMI, "
            "hypertension, heart disease, smoking history, age, gender, or "
            "diabetes outcome distribution. Input: a plain English question."
        ),
        model_name=MODEL_NAME,
    )

    web_tool = make_web_search_tool()

    tools = [heart_tool, cancer_tool, diabetes_tool, web_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    return executor


def main():
    for db in (HEART_DB, CANCER_DB, DIABETES_DB):
        if not os.path.exists(db):
            print(f"⚠️  Missing {db}. Run `python build_databases.py` first "
                  f"(after placing CSVs in ./data/ or running "
                  f"generate_sample_data.py for demo data).")
            return

    executor = build_agent()
    print(f"\n🩺 Multi-Tool Medical AI Agent  (model: {MODEL_NAME})")
    print("Ask about dataset stats (heart/cancer/diabetes) or general medical knowledge.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue
        result = executor.invoke({"input": question})
        print(f"\nAgent: {result['output']}\n")


if __name__ == "__main__":
    main()
