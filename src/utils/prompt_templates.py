# src/utils/prompt_templates.py
"""Utility module for prompt templates used in research tasks"""

from typing import Dict, Any

class PromptTemplates:
    """Collection of optimized prompt templates"""
    
    RESEARCH_SYNTHESIS = """
    You are a research assistant tasked with synthesizing information from multiple sources.
    
    Question: {question}
    
    Sources:
    {sources}
    
    Please provide a comprehensive answer that:
    1. Directly addresses the question
    2. Integrates information from multiple sources
    3. Identifies any conflicting information
    4. Acknowledges limitations in the available data
    5. Suggests areas for further investigation
    
    Format your response with clear sections and cite sources using [Source X] notation.
    
    Response:
    """
    
    DOCUMENT_ANALYSIS = """
    Analyze the following document for key insights relevant to: {topic}
    
    Document:
    {document}
    
    Please extract:
    - Main arguments or findings
    - Supporting evidence
    - Methodology (if applicable)
    - Limitations or gaps
    - Relevance to the topic
    
    Analysis:
    """
    
    CRITIQUE_GENERATION = """
    Provide a balanced critique of the following research or argument:
    
    {content}
    
    Consider:
    - Strengths of the argument/research
    - Weaknesses or limitations
    - Methodological concerns
    - Alternative perspectives
    - Suggestions for improvement
    
    Critique:
    """