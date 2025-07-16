# src/main.py
import asyncio
import typer
from rich.console import Console
from pathlib import Path
import yaml

from models.model_manager import ModelManager
from rag.vector_store import VectorStore
from rag.document_processor import DocumentProcessor
from agents.research_agent import ResearchAgent, ResearchQuery

app = typer.Typer()
console = Console()

@app.command()
def setup():
    """Setup the research agent environment"""
    console.print("[bold green]Setting up Research Agent...[/bold green]")
    
    # Load configuration
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    model_manager = ModelManager(config)
    vector_store = VectorStore()
    
    # Load base model
    model_manager.load_base_model(config["base_model"])
    
    console.print("[bold green]Setup complete![/bold green]")

@app.command()
def ingest_documents(directory: str):
    """Ingest documents into vector store"""
    console.print(f"[bold blue]Ingesting documents from {directory}...[/bold blue]")
    
    processor = DocumentProcessor()
    vector_store = VectorStore()
    
    doc_path = Path(directory)
    for file_path in doc_path.glob("**/*"):
        if file_path.suffix in processor.supported_formats:
            try:
                doc = processor.process_file(file_path)
                chunks = processor.chunk_document(doc)
                vector_store.add_documents(chunks)
                console.print(f"✓ Processed {file_path.name}")
            except Exception as e:
                console.print(f"✗ Error processing {file_path.name}: {e}")
    
    stats = vector_store.get_collection_stats()
    console.print(f"[bold green]Ingestion complete! {stats['total_documents']} documents indexed.[/bold green]")

@app.command()
def research(question: str, depth: str = "medium"):
    """Perform research query"""
    console.print(f"[bold blue]Researching: {question}[/bold blue]")
    
    # Initialize components
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_manager = ModelManager(config)
    vector_store = VectorStore()
    agent = ResearchAgent(model_manager, vector_store)
    
    # Load model
    model_manager.load_base_model(config["base_model"])
    
    # Create query
    query = ResearchQuery(question=question, depth=depth)
    
    # Process query
    async def run_research():
        return await agent.process_research_query(query)
    
    result = asyncio.run(run_research())
    
    # Display results
    console.print("\n[bold green]Research Results:[/bold green]")
    console.print(result["synthesis"])
    
    console.print("\n[bold blue]Sources:[/bold blue]")
    for source in result["sources"]:
        console.print(f"- {source}")

if __name__ == "__main__":
    app()