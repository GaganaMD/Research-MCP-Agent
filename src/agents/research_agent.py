from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime
import asyncio

# Import statements with proper error handling
try:
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
    import faiss
    import numpy as np
    import pandas as pd
except ImportError as e:
    logging.error(f"Failed to import required libraries: {e}")
    raise

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, model_name: str, model_type: str = "sentence-transformer"):
        """Load a model based on type"""
        try:
            if model_type == "sentence-transformer":
                model = SentenceTransformer(model_name)
                model.to(self.device)
                self.models[model_name] = model
                return model
            elif model_type == "transformers":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                model.to(self.device)
                self.tokenizers[model_name] = tokenizer
                self.models[model_name] = model
                return model
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def get_model(self, model_name: str):
        return self.models.get(model_name)
    
    def get_tokenizer(self, model_name: str):
        return self.tokenizers.get(model_name)
    
    def list_models(self):
        return list(self.models.keys())

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
        
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict] = None):
        """Add documents with their embeddings to the vector store"""
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar documents"""
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'distance': distances[0][i],
                    'similarity': 1 / (1 + distances[0][i])
                })
        return results
    
    def get_document_count(self):
        return len(self.documents)
    
    def clear(self):
        """Clear all documents from the vector store"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []

class ResearchAgent:
    def __init__(self):
        self.model_manager = ModelManager()
        self.vector_store = VectorStore()
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the research agent with default models"""
        try:
            # Load default embedding model
            self.model_manager.load_model("all-MiniLM-L6-v2")
            self.logger.info("Research Agent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Research Agent: {e}")
            return False
    
    async def process_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Process and store documents in the vector store"""
        try:
            model = self.model_manager.get_model("all-MiniLM-L6-v2")
            if not model:
                raise Exception("Model not loaded")
            
            # Generate embeddings
            embeddings = model.encode(documents)
            
            # Add to vector store
            self.vector_store.add_documents(documents, embeddings, metadata)
            
            self.logger.info(f"Processed {len(documents)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process documents: {e}")
            return False
    
    async def query(self, query: str, max_results: int = 5):
        """Query the vector store with a text query"""
        try:
            model = self.model_manager.get_model("all-MiniLM-L6-v2")
            if not model:
                raise Exception("Model not loaded")
            
            # Generate query embedding
            query_embedding = model.encode([query])
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k=max_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query: {e}")
            return []