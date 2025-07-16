# src/utils/optimization.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
import time
from typing import List, Dict, Any

class OptimizationUtils:
    """Utilities for performance optimization"""
    
    @staticmethod
    def optimize_model_loading():
        """Optimize model loading for production"""
        # Use torch.compile for PyTorch 2.0+
        if hasattr(torch, 'compile'):
            return torch.compile
        return lambda x: x
    
    @staticmethod
    def batch_inference(model, tokenizer, prompts: List[str], batch_size: int = 4):
        """Batch inference for better throughput"""
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode batch
            batch_results = tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True
            )
            
            results.extend(batch_results)
        
        return results
    
    @staticmethod
    def memory_efficient_attention(model):
        """Enable memory efficient attention"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Enable flash attention if available
        if hasattr(model.config, 'use_flash_attention'):
            model.config.use_flash_attention = True
    
    @staticmethod
    def optimize_vector_search(vector_store, query_embedding, k: int = 5):
        """Optimize vector similarity search"""
        # Use approximate nearest neighbor search for large datasets
        # Implementation depends on your vector store
        pass