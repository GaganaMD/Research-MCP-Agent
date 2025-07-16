from typing import List, Dict, Any, Optional
import json
import pickle
import numpy as np
import faiss
import logging
from datetime import datetime

class Document:
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}
        self.id = None
        self.embedding = None
        self.created_at = datetime.now()
    
    def to_dict(self):
        return {
            'content': self.content,
            'metadata': self.metadata,
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        doc = cls(data['content'], data.get('metadata', {}))
        doc.id = data.get('id')
        if data.get('created_at'):
            doc.created_at = datetime.fromisoformat(data['created_at'])
        return doc
    
    def __str__(self):
        return f"Document(id={self.id}, content='{self.content[:50]}...', metadata={self.metadata})"
    
    def __repr__(self):
        return self.__str__()

class VectorStoreManager:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Document] = []
        self.logger = logging.getLogger(__name__)
        
    def add_document(self, document: Document, embedding: np.ndarray):
        """Add a single document with its embedding"""
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} doesn't match expected {self.dimension}")
        
        document.id = len(self.documents)
        document.embedding = embedding
        self.documents.append(document)
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        
        self.logger.info(f"Added document with ID {document.id}")
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add multiple documents with their embeddings"""
        if embeddings.shape[0] != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match expected {self.dimension}")
        
        start_id = len(self.documents)
        
        for i, doc in enumerate(documents):
            doc.id = start_id + i
            doc.embedding = embeddings[i]
        
        self.documents.extend(documents)
        self.index.add(embeddings.astype('float32'))
        
        self.logger.info(f"Added {len(documents)} documents")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = None):
        """Search for similar documents"""
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} doesn't match expected {self.dimension}")
        
        if len(self.documents) == 0:
            return []
        
        # Ensure k doesn't exceed available documents
        k = min(k, len(self.documents))
        
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                distance = distances[0][i]
                similarity = 1 / (1 + distance)
                
                # Apply threshold if specified
                if threshold is None or similarity >= threshold:
                    results.append({
                        'document': self.documents[idx],
                        'distance': distance,
                        'similarity': similarity,
                        'rank': i + 1
                    })
        
        return results
    
    def get_document_by_id(self, doc_id: int) -> Optional[Document]:
        """Get a document by its ID"""
        if 0 <= doc_id < len(self.documents):
            return self.documents[doc_id]
        return None
    
    def get_documents_by_metadata(self, metadata_filter: Dict[str, Any]) -> List[Document]:
        """Get documents that match metadata criteria"""
        results = []
        for doc in self.documents:
            match = True
            for key, value in metadata_filter.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                results.append(doc)
        return results
    
    def update_document(self, doc_id: int, content: str = None, metadata: Dict[str, Any] = None, embedding: np.ndarray = None):
        """Update a document"""
        if doc_id < 0 or doc_id >= len(self.documents):
            raise ValueError(f"Document ID {doc_id} not found")
        
        doc = self.documents[doc_id]
        
        if content is not None:
            doc.content = content
        
        if metadata is not None:
            doc.metadata.update(metadata)
        
        if embedding is not None:
            if embedding.shape[0] != self.dimension:
                raise ValueError(f"Embedding dimension {embedding.shape[0]} doesn't match expected {self.dimension}")
            
            doc.embedding = embedding
            # Update the embedding in the FAISS index
            self.index.remove_ids(np.array([doc_id]))
            self.index.add_with_ids(embedding.reshape(1, -1).astype('float32'), np.array([doc_id]))
        
        self.logger.info(f"Updated document {doc_id}")
    
    def delete_document(self, doc_id: int):
        """Delete a document (marks as deleted, doesn't actually remove from index)"""
        if doc_id < 0 or doc_id >= len(self.documents):
            raise ValueError(f"Document ID {doc_id} not found")
        
        # Mark document as deleted
        self.documents[doc_id].metadata['_deleted'] = True
        self.logger.info(f"Marked document {doc_id} as deleted")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        active_docs = [doc for doc in self.documents if not doc.metadata.get('_deleted', False)]
        
        return {
            'total_documents': len(self.documents),
            'active_documents': len(active_docs),
            'deleted_documents': len(self.documents) - len(active_docs),
            'dimension': self.dimension,
            'index_size': self.index.ntotal,
            'metadata_keys': list(set(
                key for doc in active_docs 
                for key in doc.metadata.keys() 
                if not key.startswith('_')
            ))
        }
    
    def clear(self):
        """Clear all documents from the vector store"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.logger.info("Cleared all documents from vector store")
    
    def save(self, filepath: str):
        """Save the vector store to disk"""
        try:
            # Prepare data for serialization
            data = {
                'documents': [doc.to_dict() for doc in self.documents],
                'dimension': self.dimension,
                'version': '1.0'
            }
            
            # Save document data
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Save FAISS index
            faiss.write_index(self.index, filepath + ".index")
            
            self.logger.info(f"Vector store saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")
            return False
    
    def load(self, filepath: str):
        """Load the vector store from disk"""
        try:
            # Load document data
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Restore documents
            self.documents = [Document.from_dict(doc_data) for doc_data in data['documents']]
            self.dimension = data['dimension']
            
            # Load FAISS index
            self.index = faiss.read_index(filepath + ".index")
            
            self.logger.info(f"Vector store loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            return False
    
    def rebuild_index(self):
        """Rebuild the FAISS index from stored embeddings"""
        try:
            # Create new index
            new_index = faiss.IndexFlatL2(self.dimension)
            
            # Collect all active embeddings with their original IDs
            embeddings_to_add = []
            original_ids = []
            
            for doc in self.documents:
                if not doc.metadata.get('_deleted', False) and doc.embedding is not None:
                    embeddings_to_add.append(doc.embedding)
                    original_ids.append(doc.id) # Use original ID for FAISS
            
            if embeddings_to_add:
                embeddings_array = np.array(embeddings_to_add).astype('float32')
                ids_array = np.array(original_ids).astype('int64') # FAISS expects int64 for IDs
                new_index.add_with_ids(embeddings_array, ids_array)
            
            self.index = new_index
            self.logger.info("FAISS index rebuilt successfully.")
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild FAISS index: {e}")

