
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
- **Tavily Tool**: To search the internet.
- **LangChain**: To handle language model interactions and tools.
- **LangGraph**: To organize agent steps in a flow.
- **Streamlit**: To create a UI for users.

---

## Code Explaination

### 1. Import Tools
```python
from langchain... import TavilySearchResults, Chroma, OpenAIEmbeddings, RetrievalQA, OpenAI
from langgraph.graph import StateGraph
import streamlit as st
```
You import search tools, language models, graph management, and UI tools.

---

### 2. Set Up Tools
```python
tavily_tool = TavilySearchResults(k=5)
embedding_fn = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embedding_fn)
llm = OpenAI()
```
- Tavily will search for top 5 web results.
- Embeddings convert text to vectors (for smart search).
- VectorStore holds all found info for easy lookup.
- `llm` is the brain that answers your question.

---

### 3. Agent 1: Research Agent
```python
def research_agent(state):
    query = state["query"]
    results = tavily_tool.run(query)
    docs = [{"content": r["snippet"], "metadata": {"source": r["url"]}} for r in results]
    vectorstore.add_documents(docs)
    return {"status": "data_collected", "query": query}
```
- Takes user’s question.
- Uses Tavily to get web snippets.
- Stores those snippets in a vector database.

---

### 4. Agent 2: Answer Agent
```python
def answer_agent(state):
    query = state["query"]
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa_chain.run(query)
    return {"answer": answer}
```
- Pulls info from the database.
- Uses AI to write a complete answer.

---

### 5. LangGraph Flow
```python
graph = StateGraph()
graph.add_node("Research", research_agent)
graph.add_node("Drafting", answer_agent)
graph.set_entry_point("Research")
graph.add_edge("Research", "Drafting")
graph.set_finish_point("Drafting")
final_graph = graph.compile()
```
This builds the **2-step flow**:
1. Research → 2. Draft Answer → Done

---

### 6. Streamlit UI
```python
st.title("Deep Research AI Agent")
user_query = st.text_input("Your Research Question:")

if st.button("Run Agents") and user_query:
    input_state = {"query": user_query}
    result = final_graph.invoke(input_state)
    st.write(result["answer"])
```
You build a simple interface:
- User types a question.
- Clicks a button.
- Agents do their job and show the answer.
