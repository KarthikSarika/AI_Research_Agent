from flask import Flask, render_template, request
from langgraph.graph import StateGraph
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from pydantic import BaseModel
from typing import List
import os

app = Flask(__name__)

# State Schema
class ResearchState(BaseModel):
    topic: str
    docs: List[str] = []
    final_answer: str = ""
    search_results: List[str] = []  # To hold structured search results

# Research Agent
def research_agent(state: ResearchState) -> dict:
    print("🔍 Researching:", state.topic)
    tavily_tool = TavilySearchResults(k=5, tavily_api_key=os.getenv("TAVILY_API_KEY"))
    search_results = tavily_tool.run(state.topic)

    if not search_results:
        return {"docs": ["No valid content found."], "search_results": []}

    # Collect search result URLs
    state.search_results = [result['url'] for result in search_results[:3]]

    all_docs = []
    for url in state.search_results:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            text_transformer = Html2TextTransformer()
            plain_docs = text_transformer.transform_documents(docs)
            all_docs.extend([doc.page_content for doc in plain_docs])
        except Exception as e:
            print(f"⚠️ Failed to load {url}: {e}")

    if not all_docs:
        return {"docs": ["No valid content found."], "search_results": state.search_results}

    summaries = [doc for doc in all_docs]  # Placeholder for actual summarization
    return {"docs": summaries, "search_results": state.search_results}

# Drafting Agent
def drafting_agent(state: ResearchState) -> dict:
    print("📝 Drafting Answer...")
    merged = "\n\n".join(state.docs)
    answer = f"Here's a synthesized summary on the topic '{state.topic}':\n\n{merged}"
    return {"final_answer": answer, "search_results": state.search_results}

# LangGraph Workflow
graph = StateGraph(state_schema=ResearchState)
graph.add_node("research", research_agent)
graph.add_node("draft", drafting_agent)
graph.set_entry_point("research")
graph.add_edge("research", "draft")
graph.set_finish_point("draft")

app_compiled = graph.compile()

# Flask Route for Handling Requests
@app.route("/", methods=["GET", "POST"])
def index():
    final_answer = None
    search_results = []
    if request.method == "POST":
        query = request.form["query"]
        if query:
            try:
                result = app_compiled.invoke({"topic": query})
                final_answer = result["final_answer"]
                search_results = result["search_results"]
            except Exception as e:
                final_answer = f"⚠️ Error: {e}"

    return render_template("index.html", final_answer=final_answer, search_results=search_results)

if __name__ == "__main__":
    app.run(debug=True)
