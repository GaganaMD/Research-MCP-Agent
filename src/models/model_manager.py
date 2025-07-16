# src/models/model_manager.py
"""Model Manager for loading and managing LLMs"""


import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
from typing import Optional, Dict, Any
import logging

class ModelManager:
    """Manages loading and inference of various LLMs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        
    def load_base_model(self, model_name: str) -> None:
        """Load a base model (Qwen, LLaMA, Mistral, etc.)"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
    def load_fine_tuned_model(self, base_model: str, adapter_path: str) -> None:
        """Load a fine-tuned PEFT model"""
        if base_model not in self.models:
            self.load_base_model(base_model)
            
        peft_config = PeftConfig.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(
            self.models[base_model], 
            adapter_path
        )
        
        model_key = f"{base_model}_finetuned"
        self.models[model_key] = model
        self.tokenizers[model_key] = self.tokenizers[base_model]
        
    def generate_response(
        self, 
        model_name: str, 
        prompt: str, 
        max_length: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate response using specified model"""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()