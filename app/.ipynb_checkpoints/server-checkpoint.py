from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_core.runnables import Runnable, RunnableConfig
from app.RAG import initialize, predict

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

class LocalChain(Runnable):
    def __init__(self, data_path):
        self.model = initialize(data_path)

    async def invoke(self, input_text: str) -> str:
        return predict(self.model, input_text)

    def config(self) -> RunnableConfig:
        return RunnableConfig(
            inputs={"input_text": {"type": "str", "description": "Input text to process."}},
            outputs={"output_text": {"type": "str", "description": "Processed output text."}}
        )

data_path = '/data_2/ace_myyak/results.csv'
local_chain = LocalChain(data_path)

add_routes(app, local_chain, path="/local-chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
