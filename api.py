from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from src.rag import RAGPipeline

app = FastAPI(title="Sentinel RAG API", version="1.0.0")
rag_pipeline = RAGPipeline()


class EvidenceLawsRequest(BaseModel):
    evidencia: str = Field(..., min_length=10, max_length=5000)
    max_leyes: int = Field(3, ge=1, le=3)


@app.get("/health")
def health_check():
    return {"success": True, "message": "RAG service operativo"}


@app.get("/rag/query")
async def rag_query(
    question: str = Query(..., min_length=1, max_length=2000),
    conversation_id: str = Query("chat_principal", min_length=1, max_length=100),
):
    try:
        result = await run_in_threadpool(rag_pipeline.query, conversation_id, question)
        return {"success": True, "data": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error procesando consulta RAG: {exc}")


@app.post("/rag/evidence/laws")
async def rag_evidence_laws(payload: EvidenceLawsRequest):
    try:
        result = await run_in_threadpool(
            rag_pipeline.get_law_protections_for_evidence,
            payload.evidencia,
            payload.max_leyes,
        )
        return {"success": True, "data": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error procesando evidencia legal: {exc}")


@app.on_event("shutdown")
def on_shutdown():
    rag_pipeline.close()
