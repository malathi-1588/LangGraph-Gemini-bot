from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import ChatRequest
from llm import initialize_chat_engine
import os

# Set Gemini API Key
from dotenv import load_dotenv
load_dotenv()
#os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chat engine
chat_engine = initialize_chat_engine()

@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        response = chat_engine.invoke({"question": request.message})
        return {"response": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
