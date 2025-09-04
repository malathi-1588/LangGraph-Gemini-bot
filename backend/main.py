from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from models import ChatRequest
from llm_graph import LLMGraphApp

load_dotenv()

app = FastAPI(title="Hybrid RAG + Web Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the graph app once on startup
graph_app = LLMGraphApp(data_dir="data", index_dir="vectorstore", model_name="gemini-2.5-pro")

@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        result = graph_app.invoke(question=request.message, session_id=request.session_id or "default")
        return {"response": result["answer"], "route": result["route"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
