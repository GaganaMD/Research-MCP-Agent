# tests/test_agents.py
import pytest
import asyncio
from unittest.mock import Mock, patch
from src.agents.research_agent import ResearchAgent, ResearchQuery
from src.models.model_manager import ModelManager
from src.rag.vector_store import VectorStore

class TestResearchAgent:
    
    @pytest.fixture
    def mock_model_manager(self):
        manager = Mock(spec=ModelManager)
        manager.generate_response.return_value = "Test response"
        return manager
    
    @pytest.fixture
    def mock_vector_store(self):
        store = Mock(spec=VectorStore)
        store.similarity_search.return_value = [
            {"content": "Test content", "metadata": {}, "score": 0.9}
        ]
        return store
    
    @pytest.fixture
    def research_agent(self, mock_model_manager, mock_vector_store):
        return ResearchAgent(mock_model_manager, mock_vector_store)
    
    @pytest.mark.asyncio
    async def test_document_search(self, research_agent):
        query = ResearchQuery(question="What is machine learning?")
        results = await research_agent._search_documents(query.question)
        assert len(results) > 0
        assert "content" in results[0]
    
    @pytest.mark.asyncio
    async def test_research_workflow(self, research_agent):
        query = ResearchQuery(question="What is deep learning?", depth="medium")
        result = await research_agent.process_research_query(query)
        
        assert "synthesis" in result
        assert "sources" in result
        assert "workflow_steps" in result
        assert len(result["workflow_steps"]) >= 1

# tests/test_rag.py
import pytest
from pathlib import Path
from src.rag.document_processor import DocumentProcessor, Document
from src.rag.vector_store import VectorStore
import tempfile
import os

class TestDocumentProcessor:
    
    @pytest.fixture
    def processor(self):
        return DocumentProcessor()
    
    def test_text_processing(self, processor):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.")
            f.flush()
            
        try:
            doc = processor.process_file(Path(f.name))
            assert doc.content == "This is a test document."
            assert doc.source == f.name
        finally:
            os.unlink(f.name)
    
    def test_document_chunking(self, processor):
        doc = Document(
            content="A" * 2000,
            metadata={},
            source="test.txt"
        )
        chunks = processor.chunk_document(doc, chunk_size=500)
        assert len(chunks) == 4
        assert all(len(chunk.content) <= 500 for chunk in chunks)

class TestVectorStore:
    
    @pytest.fixture
    def vector_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield VectorStore(persist_directory=temp_dir)
    
    def test_add_and_search(self, vector_store):
        docs = [
            Document("Machine learning is a subset of AI", {}, "test1.txt"),
            Document("Deep learning uses neural networks", {}, "test2.txt")
        ]
        
        vector_store.add_documents(docs)
        results = vector_store.similarity_search("artificial intelligence", k=1)
        
        assert len(results) == 1
        assert "machine learning" in results[0]["content"].lower()

# tests/test_models.py
import pytest
from unittest.mock import Mock, patch
from src.models.model_manager import ModelManager
from src.models.fine_tuner import LLMFineTuner

class TestModelManager:
    
    @pytest.fixture
    def config(self):
        return {
            "base_model": "microsoft/DialoGPT-small",
            "device": "cpu"
        }
    
    @patch('src.models.model_manager.AutoModelForCausalLM')
    @patch('src.models.model_manager.AutoTokenizer')
    def test_model_loading(self, mock_tokenizer, mock_model, config):
        manager = ModelManager(config)
        manager.load_base_model("microsoft/DialoGPT-small")
        
        assert "microsoft/DialoGPT-small" in manager.models
        assert "microsoft/DialoGPT-small" in manager.tokenizers
    
    def test_model_manager_init(self, config):
        manager = ModelManager(config)
        assert manager.config == config
        assert manager.models == {}
        assert manager.tokenizers == {}