# scripts/build_vector_store.py
import argparse
from pathlib import Path
import logging
from rich.progress import Progress, SpinnerColumn, TextColumn
from src.rag.document_processor import DocumentProcessor
from src.rag.vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Build vector store from documents")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output-dir", type=str, default="./data/embeddings", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize components
    processor = DocumentProcessor()
    vector_store = VectorStore(persist_directory=args.output_dir)
    
    input_path = Path(args.input_dir)
    supported_files = []
    
    # Find all supported files
    for file_path in input_path.rglob("*"):
        if file_path.suffix.lower() in processor.supported_formats:
            supported_files.append(file_path)
    
    logging.info(f"Found {len(supported_files)} supported files")
    
    # Process files with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        
        task = progress.add_task("Processing documents...", total=len(supported_files))
        
        for file_path in supported_files:
            try:
                # Process document
                doc = processor.process_file(file_path)
                chunks = processor.chunk_document(doc, chunk_size=args.chunk_size)
                
                # Add to vector store
                vector_store.add_documents(chunks)
                
                progress.update(task, advance=1, description=f"Processed {file_path.name}")
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                continue
    
    # Get final stats
    stats = vector_store.get_collection_stats()
    logging.info(f"Vector store built successfully!")
    logging.info(f"Total documents: {stats['total_documents']}")
    logging.info(f"Embedding dimension: {stats['embedding_dimension']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()