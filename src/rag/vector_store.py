# src/rag/vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

class VectorStore:
    """Manages document embeddings and retrieval"""
    
    def __init__(self, persist_directory: str = "./data/embeddings"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection("research_docs")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_criteria: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            where=filter_criteria
        )
        
        return [
            {
                "content": doc,
                "metadata": meta,
                "score": score
            }
            for doc, meta, score in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_documents": self.collection.count(),
            "embedding_dimension": len(self.embedding_model.encode(["test"])[0])
        }