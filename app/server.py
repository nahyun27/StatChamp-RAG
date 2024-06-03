from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Tuple
from app.RAG import initialize, predict

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

# RAG 모델 초기화
data_path = '/data_2/ace_myyak/my-demo/data/stats.csv'
rag_model = initialize(data_path)

@app.post("/local-chain")
async def run_local_chain(input_data: ChatHistory):
    try:
        print(f"DEBUG: input_data = {input_data}")  # 디버그 메시지 추가
        result = predict(rag_model, input_data.chat_history, input_data.question)
        print(f"DEBUG: result = {result}")  # 디버그 메시지 추가
        return JSONResponse(content={
            "standalone_answer": result["standalone_answer"],
            "final_answer": result["final_answer"]
        })
    except Exception as e:
        print(f"ERROR: {e}")  # 오류 메시지 출력
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
