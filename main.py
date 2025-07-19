from flask import Flask, render_template, request
from langgraph.graph import StateGraph
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Flask app
app = Flask(__name__)

# Define LangGraph State
class ResearchState(BaseModel):
    topic: str
    docs: List[str] = []
    final_answer: str = ""
    search_results: List[str] = []

# Research Node
def research_agent(state: ResearchState) -> dict:
    print(f"🔍 Researching topic: {state.topic}")
    
    # Use Tavily search tool (load API key from env)
    tavily_tool = TavilySearchResults(k=5, tavily_api_key=os.environ.get("TAVILY_API_KEY"))
    search_results = tavily_tool.run(state.topic)

    if not search_results:
        print("⚠️ No search results returned.")
        return {"docs": ["No valid content found."], "search_results": []}

    state.search_results = [result['url'] for result in search_results[:5]]
    all_docs = []

    for url in state.search_results:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            transformer = Html2TextTransformer()
            plain_docs = transformer.transform_documents(docs)
            all_docs.extend([doc.page_content for doc in plain_docs])
        except Exception as e:
            print(f"⚠️ Error loading {url}: {e}")

    if not all_docs:
        return {"docs": ["No valid content extracted."], "search_results": state.search_results}

    return {"docs": all_docs, "search_results": state.search_results}

# Draft Node
def drafting_agent(state: ResearchState) -> dict:
    print("📝 Drafting summary...")
    merged = "\n\n".join(state.docs)

    prompt = f"Please synthesize the following content into a structured summary (max 200 words) on the topic '{state.topic}':\n\n{merged}"

    try:
        response = model.generate_content(prompt)
        summary = response.text
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        summary = "Could not generate a summary due to an internal error."

    return {
        "final_answer": f"Here's a synthesized summary on the topic '{state.topic}':\n\n{summary}",
        "search_results": state.search_results
    }

# Build LangGraph workflow
graph = StateGraph(state_schema=ResearchState)
graph.add_node("research", research_agent)
graph.add_node("draft", drafting_agent)
graph.set_entry_point("research")
graph.add_edge("research", "draft")
graph.set_finish_point("draft")

app_compiled = graph.compile()

# Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    final_answer = None
    search_results = []
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            try:
                result = app_compiled.invoke({"topic": query})
                final_answer = result["final_answer"]
                search_results = result["search_results"]
            except Exception as e:
                print(f"❌ Error during workflow: {e}")
                final_answer = f"⚠️ Error: {e}"
    return render_template("index.html", final_answer=final_answer, search_results=search_results)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
