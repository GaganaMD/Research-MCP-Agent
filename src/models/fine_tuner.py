# src/models/fine_tuner.py
"""Fine-tuner for LLMs using PEFT/LoRA for research tasks"""

import torch
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
from typing import List, Dict

class LLMFineTuner:
    """Fine-tune LLMs using PEFT/LoRA for research tasks"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
    def prepare_training_data(self, data_path: str) -> Dataset:
        """Prepare dataset for fine-tuning"""
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        def format_prompt(example):
            return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
            
        formatted_data = [format_prompt(item) for item in data]
        return Dataset.from_dict({"text": formatted_data})
        
    def setup_lora_config(self) -> LoraConfig:
        """Configure LoRA parameters"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
    def fine_tune_model(
        self, 
        base_model_name: str, 
        training_data: Dataset,
        output_dir: str
    ) -> None:
        """Fine-tune model with LoRA"""
        model = self.model_manager.models[base_model_name]
        tokenizer = self.model_manager.tokenizers[base_model_name]
        
        # Apply LoRA
        lora_config = self.setup_lora_config()
        model = get_peft_model(model, lora_config)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512
            )
            
        tokenized_dataset = training_data.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=100,
            warmup_steps=100,
            fp16=True,
            dataloader_pin_memory=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )
        
        # Train
        trainer.train()
        trainer.save_model()