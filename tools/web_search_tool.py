"""
web_search_tool.py

MedicalWebSearchTool: for general medical knowledge (definitions, symptoms,
causes, treatments/cures) — NOT for dataset statistics.

Uses DuckDuckGo search via langchain_community (free, no API key required).

Optional: swap in Tavily for more reliable results by setting TAVILY_API_KEY
in your .env and uncommenting the Tavily block below.
"""

import os
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# --- Optional Tavily alternative (uncomment if you have a TAVILY_API_KEY) ---
# from langchain_community.tools.tavily_search import TavilySearchResults
#
# def make_web_search_tool() -> Tool:
#     search = TavilySearchResults(max_results=4)
#     return Tool(
#         name="MedicalWebSearchTool",
#         description=(
#             "Use this tool ONLY for general medical knowledge questions such as "
#             "definitions, symptoms, causes, treatments, or cures of a disease. "
#             "Do NOT use this for statistics or numeric queries about the "
#             "datasets — use the DB tools for that."
#         ),
#         func=lambda q: str(search.invoke(q)),
#     )

def make_web_search_tool() -> Tool:
    search = DuckDuckGoSearchRun()
    return Tool(
        name="MedicalWebSearchTool",
        description=(
            "Use this tool ONLY for general medical knowledge questions such as "
            "definitions, symptoms, causes, treatments, or cures of a disease "
            "(e.g. 'What is diabetes?', 'What are the symptoms of lung cancer?', "
            "'How is heart disease treated?'). "
            "Do NOT use this for statistics, counts, averages, or numeric "
            "queries about the datasets — use HeartDiseaseDBTool, CancerDBTool, "
            "or DiabetesDBTool for that instead. Input should be a search query string."
        ),
        func=search.run,
    )
