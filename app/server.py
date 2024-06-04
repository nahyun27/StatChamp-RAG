# app/server.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from typing import List, Tuple
from app.RAG import initialize, predict

        # What is the barrel percent of the Kyle Tucker in 2024?
        # How is the performance of Heungmin Son in 2024?
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.get("/local-chain/playground")
async def redirect_playground():
    return RedirectResponse("/local-chain/playground/")

@app.get("/local-chain/playground/")
async def get_playground(request: Request):
    return templates.TemplateResponse("playground.html", {"request": request})

class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str
    sport: str

# 모델 초기화
baseball_data_path = 'data/stats.csv'
soccer_players_data_path = 'data/premier_league_players.csv'
soccer_stats_data_path = 'data/premier_league_players_stats_2324.csv'
soccer_stats_data_entire_path = 'data/premier_league_players_stats.csv'

baseball_model = initialize(baseball_data_path, soccer_players_data_path, soccer_stats_data_path, soccer_stats_data_entire_path, 'baseball')
soccer_model = initialize(baseball_data_path, soccer_players_data_path, soccer_stats_data_path, soccer_stats_data_entire_path, 'soccer')

@app.post("/local-chain")
async def run_local_chain(input_data: ChatHistory):
    try:
        print(f"DEBUG: input_data = {input_data}")
        if input_data.sport == 'baseball':
            model = baseball_model
        elif input_data.sport == 'soccer':
            model = soccer_model
        else:
            raise HTTPException(status_code=400, detail="Invalid sport type")

        results = predict(model, input_data.chat_history, input_data.question)
        return JSONResponse(content=results)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
