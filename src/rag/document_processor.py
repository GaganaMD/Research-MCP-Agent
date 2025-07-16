# src/rag/document_processor.py
"""Document Processor for RAG tasks"""

import pypdf
import docx
from pathlib import Path
from typing import List, Dict, Any
import logging
from dataclasses import dataclass

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    source: str

class DocumentProcessor:
    """Process various document types for RAG"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt', '.docx', '.md'}
        
    def process_file(self, file_path: Path) -> Document:
        """Process a single file"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self._process_pdf(file_path)
        elif suffix == '.txt' or suffix == '.md':
            return self._process_text(file_path)
        elif suffix == '.docx':
            return self._process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
            
    def _process_pdf(self, file_path: Path) -> Document:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
                
        return Document(
            content=text,
            metadata={"page_count": len(pdf_reader.pages)},
            source=str(file_path)
        )
        
    def _process_text(self, file_path: Path) -> Document:
        """Process plain text files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            
        return Document(
            content=text,
            metadata={"file_size": file_path.stat().st_size},
            source=str(file_path)
        )
        
    def _process_docx(self, file_path: Path) -> Document:
        """Process Word documents"""
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        return Document(
            content=text,
            metadata={"paragraph_count": len(doc.paragraphs)},
            source=str(file_path)
        )
        
    def chunk_document(self, document: Document, chunk_size: int = 1000) -> List[Document]:
        """Split document into chunks"""
        chunks = []
        text = document.content
        
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            chunks.append(Document(
                content=chunk_text,
                metadata={**document.metadata, "chunk_id": i // chunk_size},
                source=document.source
            ))
            
        return chunks