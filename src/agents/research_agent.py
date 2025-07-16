# src/agents/research_agent.py
"""Main MCP agent for research tasks"""

import asyncio
from typing import Dict, List, Any, Optional
from mcp import ClientSession, StdioServerTransport
from mcp.types import Tool, TextContent, CallToolResult
import json
from dataclasses import dataclass

@dataclass
class ResearchQuery:
    question: str
    context: Optional[str] = None
    sources: List[str] = None
    depth: str = "medium"  # shallow, medium, deep

class ResearchAgent:
    """Main MCP agent for research tasks"""
    
    def __init__(self, model_manager: ModelManager, vector_store: VectorStore):
        self.model_manager = model_manager
        self.vector_store = vector_store
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> List[Tool]:
        """Initialize available tools"""
        return [
            Tool(
                name="search_documents",
                description="Search through uploaded research documents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="web_search",
                description="Search the web for additional information",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer", "default": 3}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="analyze_text",
                description="Analyze text using fine-tuned model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "analysis_type": {"type": "string", "enum": ["summarize", "critique", "extract_key_points"]}
                    },
                    "required": ["text", "analysis_type"]
                }
            )
        ]
        
    async def process_research_query(self, query: ResearchQuery) -> Dict[str, Any]:
        """Process a research query using agentic workflow"""
        workflow_steps = []
        
        # Step 1: Document Search
        doc_results = await self._search_documents(query.question)
        workflow_steps.append({
            "step": "document_search",
            "results": doc_results
        })
        
        # Step 2: Web Search (if needed)
        if query.depth in ["medium", "deep"]:
            web_results = await self._web_search(query.question)
            workflow_steps.append({
                "step": "web_search", 
                "results": web_results
            })
        
        # Step 3: Synthesis
        synthesis = await self._synthesize_findings(query, workflow_steps)
        
        return {
            "query": query.question,
            "workflow_steps": workflow_steps,
            "synthesis": synthesis,
            "sources": self._extract_sources(workflow_steps)
        }
        
    async def _search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search local document collection"""
        results = self.vector_store.similarity_search(query, k=5)
        return results
        
    async def _web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search"""
        # Implementation depends on your web search tool
        # This is a placeholder
        return [{"content": "Web search result", "url": "https://example.com"}]
        
    async def _synthesize_findings(self, query: ResearchQuery, workflow_steps: List[Dict]) -> str:
        """Synthesize findings using LLM"""
        context = self._build_context(workflow_steps)
        
        prompt = f"""
        Based on the following research findings, provide a comprehensive answer to the question: {query.question}
        
        Context:
        {context}
        
        Please provide a well-structured response with:
        1. Direct answer to the question
        2. Supporting evidence
        3. Potential limitations or gaps
        4. Suggestions for further research
        
        Answer:
        """
        
        response = self.model_manager.generate_response(
            "qwen2.5-14b-instruct",  # or your preferred model
            prompt,
            max_length=1024,
            temperature=0.3
        )
        
        return response
        
    def _build_context(self, workflow_steps: List[Dict]) -> str:
        """Build context from workflow results"""
        context_parts = []
        
        for step in workflow_steps:
            if step["step"] == "document_search":
                context_parts.append("Document Sources:")
                for result in step["results"]:
                    context_parts.append(f"- {result['content'][:500]}...")
                    
            elif step["step"] == "web_search":
                context_parts.append("Web Sources:")
                for result in step["results"]:
                    context_parts.append(f"- {result['content'][:500]}...")
                    
        return "\n".join(context_parts)
        
    def _extract_sources(self, workflow_steps: List[Dict]) -> List[str]:
        """Extract source references"""
        sources = []
        for step in workflow_steps:
            for result in step["results"]:
                if "source" in result:
                    sources.append(result["source"])
                elif "url" in result:
                    sources.append(result["url"])
        return list(set(sources))