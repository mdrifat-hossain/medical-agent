"""
db_tool.py

Generic factory that builds a LangChain Tool wrapping a SQLite database.
Each tool:
  1. Takes a natural-language question
  2. Uses the local Ollama LLM to generate a SQLite SELECT query against
     the DB's schema
  3. Executes the query (read-only, SELECT-only for safety)
  4. Uses the LLM again to turn the raw rows into a natural-language answer
"""

import re
import sqlite3

from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def get_schema(db_path: str) -> str:
    """Introspect the SQLite DB and return a human-readable schema string."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    parts = []
    for t in tables:
        cur.execute(f'PRAGMA table_info("{t}");')
        cols = cur.fetchall()  # (cid, name, type, notnull, dflt, pk)
        col_desc = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
        parts.append(f"Table '{t}': {col_desc}")
    conn.close()
    return "\n".join(parts)


def run_sql(db_path: str, sql: str, row_limit: int = 50):
    """Execute a SELECT-only query safely and return (columns, rows)."""
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT statements are allowed for safety.")
    if re.search(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|ATTACH)\b", sql, re.IGNORECASE):
        raise ValueError("Query contains a disallowed keyword.")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(sql)
    columns = [d[0] for d in cur.description]
    rows = cur.fetchmany(row_limit)
    conn.close()
    return columns, rows


SQL_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert SQLite query generator.\n"
     "Given a database schema and a natural language question, write ONE "
     "valid SQLite SELECT query that answers the question as precisely as "
     "possible.\n"
     "Rules:\n"
     "- Output ONLY the raw SQL query. No explanation. No markdown fences.\n"
     "- Only use SELECT statements — never modify data.\n"
     "- Use aggregate functions (COUNT, AVG, MIN, MAX, etc.) when the "
     "question asks for statistics.\n\n"
     "Schema:\n{schema}"),
    ("human", "{question}"),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful medical data assistant. Given a user's question, "
     "the SQL query that was run against the dataset, and the resulting "
     "rows, answer the question in clear, concise natural language. "
     "Cite the actual numbers from the results. Never invent data that "
     "isn't present in the results. If the rows are empty, say so plainly."),
    ("human",
     "Question: {question}\n\nSQL used: {sql}\n\nColumns: {columns}\n\nRows: {rows}"),
])


def make_db_tool(db_path: str, tool_name: str, description: str, model_name: str) -> Tool:
    llm = ChatOllama(model=model_name, temperature=0)
    schema = get_schema(db_path)

    def _run(question: str) -> str:
        try:
            sql_response = llm.invoke(
                SQL_GEN_PROMPT.format_messages(schema=schema, question=question)
            )
            sql = sql_response.content.strip()
            # strip accidental markdown fences
            sql = re.sub(r"^```sql\s*|^```\s*|```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()

            columns, rows = run_sql(db_path, sql)

            summary_response = llm.invoke(
                SUMMARY_PROMPT.format_messages(
                    question=question, sql=sql, columns=columns, rows=rows
                )
            )
            return summary_response.content
        except Exception as e:
            return f"[{tool_name} error] {e}"

    return Tool(name=tool_name, description=description, func=_run)
