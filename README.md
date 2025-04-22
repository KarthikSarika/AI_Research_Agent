
# Deep Research AI Agentic System

## Problem
We often want **detailed answers** to complex questions from the internet. But searching, reading multiple articles, summarizing, and writing a clean response takes time.

---

## Goal
Create an **AI system** with:
- One agent to **search and collect information** from the internet.
- Another agent to **analyze that info and write an answer**.
- A simple **web UI** using Streamlit so users can type a question and get an answer.

---

## Solution Overview
You used:
- **Tavily Tool**: To perform real-time web searches.
- **LangChain**: To handle language model interactions and tools.
- **LangGraph**: To organize agent steps in a flow.
- **Flask**: To create a lightweight web app for user interaction.

---

## Code Explaination

### 1. Import Tools
```python
from flask import Flask, render_template, request
from langgraph.graph import StateGraph
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from pydantic import BaseModel
from typing import List

```
You import web framework components, graph management, search tools, document loaders, and schema validation.

---

### 2. Set Up Flask
```python
app = Flask(__name__)

```

## 3.Define State Schema
```python
class ResearchState(BaseModel):
    topic: str
    docs: List[str] = []
    final_answer: str = ""
    search_results: List[str] = []

```
-Defines the shared data structure across agents.

Keeps track of:
-Search topic.
-Collected documents.
-Final summarized answer.
-URLs used for research.

---

### 4. Agent 1: Research Agent
```python
def research_agent(state: ResearchState) -> dict:
    tavily_tool = TavilySearchResults(k=5, tavily_api_key="...")
    search_results = tavily_tool.run(state.topic)
    
    state.search_results = [result['url'] for result in search_results[:3]]
    all_docs = []
    for url in state.search_results:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_transformer = Html2TextTransformer()
        plain_docs = text_transformer.transform_documents(docs)
        all_docs.extend([doc.page_content for doc in plain_docs])
        
    return {"docs": all_docs or ["No valid content found."], "search_results": state.search_results}

```
-Searches the web for the topic.
-Loads the top 3 web pages.
-Converts HTML content to plain text.
-Collects all readable text into documents.

---

### 5. Agent 2: Drafting Agent
```python
def drafting_agent(state: ResearchState) -> dict:
    merged = "\n\n".join(state.docs)
    answer = f"Here's a synthesized summary on the topic '{state.topic}':\n\n{merged}"
    return {"final_answer": answer, "search_results": state.search_results}

```
- Merges all documents.
- Creates a final summarized answer for the user.

---

### 6. LangGraph Flow
```python
graph = StateGraph(state_schema=ResearchState)
graph.add_node("research", research_agent)
graph.add_node("draft", drafting_agent)
graph.set_entry_point("research")
graph.add_edge("research", "draft")
graph.set_finish_point("draft")
app_compiled = graph.compile()
```
This builds the **2-step flow**:
1. Research → 2. Draft Answer → Done

---

### 7. Flask Web UI
```python
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        if query:
            result = app_compiled.invoke({"topic": query})
            return render_template("index.html", final_answer=result["final_answer"], search_results=result["search_results"])
    return render_template("index.html")
```
You build a simple interface:
- Users type a query into a form.
- The system runs the research and drafting agents.
- The final answer and source links are shown on the page.
