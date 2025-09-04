import os
from typing import TypedDict, Literal, List, Optional

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.schema import HumanMessage, AIMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------
# State Definition
# ---------------------------

class GraphState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    route: Literal["rag", "web"]
    docs: Optional[List[str]]    # text chunks from RAG
    web: Optional[str]           # concatenated web results
    answer: Optional[str]


# ---------------------------
# Index / Retriever setup
# ---------------------------

def build_or_load_vectorstore(data_dir: str = "data", index_dir: str = "vectorstore"):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    # If a FAISS index exists, load it; otherwise build
    index_path = os.path.join(index_dir, "faiss_index")
    store_path = os.path.join(index_dir, "faiss_store.pkl")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore

    # Build from data/*.txt
    all_docs = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, fname), encoding="utf-8")
            all_docs.extend(loader.load())

    if not all_docs:
        # Create a tiny placeholder doc so the retriever doesn't crash when empty
        from langchain.docstore.document import Document
        all_docs = [Document(page_content="This is an empty knowledge base. Add .txt files to /data.")]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_dir)
    return vectorstore


# ---------------------------
# Core Components
# ---------------------------

def make_llm(model_name: str = "gemini-2.5-flash"):
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)

def make_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 6})

def make_web_tool():
    # Returns list[dict] with 'content'/'url' etc.
    return TavilySearchResults(max_results=5)


# ---------------------------
# Router (LLM-based)
# ---------------------------

ROUTER_SYSTEM = (
    "You are a router that decides whether to use local docs (RAG) or Web Search.\n"
    "Return ONLY one word: 'rag' or 'web'.\n\n"
    "Use 'rag' when the question is likely answered by internal/company/local knowledge.\n"
    "Use 'web' when the user asks for news, latest info, general facts not in local docs, "
    "statistics, dates, or anything time-sensitive or broad.\n"
)

def router_node(state: GraphState, llm: ChatGoogleGenerativeAI) -> GraphState:
    question = state["question"]
    messages = [
        HumanMessage(content=f"{ROUTER_SYSTEM}\n\nUser question: {question}")
    ]
    route = llm.invoke(messages).content.strip().lower()
    route = "web" if route not in ("rag", "web") else route
    state["route"] = route
    return state


# ---------------------------
# RAG branch
# ---------------------------

def rag_node(state: GraphState, retriever) -> GraphState:
    question = state["question"]
    docs = retriever.get_relevant_documents(question)
    state["docs"] = [d.page_content for d in docs]
    return state


# ---------------------------
# Web branch
# ---------------------------

def web_node(state: GraphState, web_tool: TavilySearchResults) -> GraphState:
    question = state["question"]
    results = web_tool.invoke({"query": question})
    # Concatenate a compact context for the final LLM
    snippets = []
    for r in results:
        # r has keys: 'content', 'url', 'title' (varies), keep it concise
        content = r.get("content") or ""
        url = r.get("url") or ""
        title = r.get("title") or ""
        snippets.append(f"Title: {title}\nURL: {url}\nSummary: {content}\n")
    state["web"] = "\n---\n".join(snippets[:5]) if snippets else "No web results."
    return state


# ---------------------------
# Compose Final Answer
# ---------------------------

FINAL_SYSTEM = (
    "You are a helpful AI assistant. Use the provided CONTEXT to answer the user.\n"
    "If the context is insufficient, say what else you would search for.\n"
    "Cite sources as plain text (titles/URLs) when using web context.\n"
    "Be concise, accurate, and avoid fabrications."
)

def answer_node(state: GraphState, llm: ChatGoogleGenerativeAI) -> GraphState:
    question = state["question"]
    chat_history = state.get("chat_history", [])

    context_blocks = []
    if state.get("docs"):
        context_blocks.append("LOCAL DOCS:\n" + "\n\n".join(state["docs"]))
    if state.get("web"):
        context_blocks.append("WEB RESULTS:\n" + state["web"])

    context_text = "\n\n".join(context_blocks) if context_blocks else "No additional context."

    messages: List[BaseMessage] = []
    messages.append(HumanMessage(content=f"{FINAL_SYSTEM}"))
    messages.extend(chat_history)
    messages.append(HumanMessage(content=f"CONTEXT:\n{context_text}\n\nUSER QUESTION:\n{question}"))

    completion = llm.invoke(messages)
    state["answer"] = completion.content
    # Append this turn to chat history
    state["chat_history"] = chat_history + [HumanMessage(content=question), AIMessage(content=state["answer"])]
    return state


# ---------------------------
# Graph Assembly
# ---------------------------

class LLMGraphApp:
    def __init__(self, data_dir: str = "data", index_dir: str = "vectorstore", model_name: str = "gemini-2.5-pro"):
        self.vectorstore = build_or_load_vectorstore(data_dir=data_dir, index_dir=index_dir)
        self.retriever = make_retriever(self.vectorstore)
        self.web_tool = make_web_tool()
        self.llm = make_llm(model_name)

        graph = StateGraph(GraphState)

        # Nodes
        graph.add_node("router", lambda s: router_node(s, self.llm))
        graph.add_node("rag_node", lambda s: rag_node(s, self.retriever))
        graph.add_node("web_node", lambda s: web_node(s, self.web_tool))
        graph.add_node("answer_node", lambda s: answer_node(s, self.llm))

        # Edges
        def route_edge(state: GraphState):
            return state["route"]

        graph.add_edge(START, "router")
        graph.add_conditional_edges("router", route_edge, {"rag": "rag_node", "web": "web_node"})
        graph.add_edge("rag_node", "answer_node")
        graph.add_edge("web_node", "answer_node")
        graph.add_edge("answer_node", END)

        # Memory (threaded)
        self.checkpointer = MemorySaver()
        self.app = graph.compile(checkpointer=self.checkpointer)

    def invoke(self, question: str, session_id: str = "default"):
        # Each session_id gets its own memory thread
        config = {"configurable": {"thread_id": session_id}}
        initial_state: GraphState = {
            "question": question,
            "chat_history": [],
            "route": "rag",
            "docs": None,
            "web": None,
            "answer": None,
        }
        result = self.app.invoke(initial_state, config=config)
        return {
            "answer": result.get("answer", ""),
            "route": result.get("route", "rag"),
        }
