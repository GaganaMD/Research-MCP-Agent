# src/api/server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from contextlib import asynccontextmanager

from src.agents.research_agent import ResearchAgent, ResearchQuery
from src.models.model_manager import ModelManager
from src.rag.vector_store import VectorStore
from src.utils.monitoring import PerformanceMonitor

# Global components
components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = {"base_model": "Qwen/Qwen2.5-14B-Instruct"}
    
    components["model_manager"] = ModelManager(config)
    components["vector_store"] = VectorStore()
    components["monitor"] = PerformanceMonitor()
    
    # Load base model
    components["model_manager"].load_base_model(config["base_model"])
    
    components["agent"] = ResearchAgent(
        components["model_manager"],
        components["vector_store"]
    )
    
    yield
    
    # Shutdown cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="MCP Research Agent API",
    description="Intelligent research assistant with RAG and LLM capabilities",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ResearchRequest(BaseModel):
    question: str
    context: Optional[str] = None
    depth: str = "medium"
    sources: Optional[List[str]] = None

class ResearchResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    workflow_steps: List[dict]
    processing_time: float

@app.post("/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    """Main research endpoint"""
    try:
        agent = components["agent"]
        monitor = components["monitor"]
        
        query = ResearchQuery(
            question=request.question,
            context=request.context,
            depth=request.depth,
            sources=request.sources or []
        )
        
        # Process with monitoring
        start_time = time.time()
        result = await agent.process_research_query(query)
        processing_time = time.time() - start_time
        
        return ResearchResponse(
            question=request.question,
            answer=result["synthesis"],
            sources=result["sources"],
            workflow_steps=result["workflow_steps"],
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    monitor = components["monitor"]
    return monitor.get_average_metrics()

@app.post("/ingest")
async def ingest_documents(background_tasks: BackgroundTasks):
    """Ingest new documents"""
    # Implementation for document ingestion
    background_tasks.add_task(ingest_documents_task)
    return {"message": "Document ingestion started"}

async def ingest_documents_task():
    """Background task for document ingestion"""
    # Implementation here
    pass