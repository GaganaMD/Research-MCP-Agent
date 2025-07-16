from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import datetime
from typing import List, Dict, Any, Optional
import logging
import json
import asyncio

# Import the fixed classes
# from research_agent import ModelManager, VectorStore, ResearchAgent
from ..agents.research_agent import ModelManager, VectorStore, ResearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Research Agent API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
research_agent = ResearchAgent()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    max_results: int = 5

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time: float
    document_count: int

class DocumentRequest(BaseModel):
    documents: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None

class DocumentResponse(BaseModel):
    success: bool
    message: str
    processed_count: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    torch_available: bool
    document_count: int
    models_loaded: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting up Research Agent API...")
    success = await research_agent.initialize()
    if not success:
        logger.error("Failed to initialize Research Agent")
        raise RuntimeError("Failed to initialize Research Agent")
    logger.info("Research Agent API started successfully")

@app.post("/documents", response_model=DocumentResponse)
async def add_documents(request: DocumentRequest):
    """Add documents to the vector store"""
    try:
        success = await research_agent.process_documents(
            request.documents, 
            request.metadata
        )
        
        if success:
            return DocumentResponse(
                success=True,
                message="Documents processed successfully",
                processed_count=len(request.documents)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to process documents")
            
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using semantic search"""
    start_time = time.time()
    
    try:
        results = await research_agent.query(request.query, request.max_results)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            results=results,
            processing_time=processing_time,
            document_count=research_agent.vector_store.get_document_count()
        )
        
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store"""
    try:
        research_agent.vector_store.clear()
        return {"success": True, "message": "All documents cleared"}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        import torch
        torch_available = torch.cuda.is_available()
    except ImportError:
        torch_available = False
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.datetime.now().isoformat(),
        torch_available=torch_available,
        document_count=research_agent.vector_store.get_document_count(),
        models_loaded=research_agent.model_manager.list_models()
    )

@app.get("/models")
async def list_models():
    """List loaded models"""
    return {
        "loaded_models": research_agent.model_manager.list_models(),
        "available_models": [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "distilbert-base-uncased"
        ]
    }

@app.post("/models/{model_name}")
async def load_model(model_name: str, model_type: str = "sentence-transformer"):
    """Load a specific model"""
    try:
        model = research_agent.model_manager.load_model(model_name, model_type)
        if model:
            return {"success": True, "message": f"Model {model_name} loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")