import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    from datasets import Dataset as HFDataset
    import pandas as pd
    import numpy as np
except ImportError as e:
    logging.error(f"Failed to import required libraries: {e}")
    raise

# from research_agent import ModelManager
from ..agents.research_agent import ModelManager, VectorStore, ResearchAgent

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FineTuner:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def prepare_data(self, texts: List[str], labels: List[int], test_size: float = 0.2):
        """Prepare data for training"""
        try:
            # Convert to pandas DataFrame for easier manipulation
            df = pd.DataFrame({'text': texts, 'label': labels})
            
            # Split data
            train_df = df.sample(frac=1-test_size, random_state=42)
            eval_df = df.drop(train_df.index)
            
            # Create datasets
            train_dataset = CustomDataset(
                train_df['text'].tolist(),
                train_df['label'].tolist(),
                self.tokenizer
            )
            
            eval_dataset = CustomDataset(
                eval_df['text'].tolist(),
                eval_df['label'].tolist(),
                self.tokenizer
            ) if len(eval_df) > 0 else None
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise
    
    def train(self, train_dataset, eval_dataset=None, num_epochs=3, output_dir="./results"):
        """Train the model"""
        try:
            # Get number of unique labels
            if hasattr(train_dataset, 'labels'):
                num_labels = len(set(train_dataset.labels))
            else:
                # Fallback: iterate through dataset to get labels
                labels = []
                for i in range(len(train_dataset)):
                    labels.append(train_dataset[i]['labels'].item())
                num_labels = len(set(labels))
            
            # Initialize model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
            
            # Resize token embeddings if needed
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                evaluation_strategy="epoch" if eval_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                greater_is_better=False,
                save_total_limit=2,
                dataloader_pin_memory=False,
                no_cuda=not torch.cuda.is_available(),
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(
                self.tokenizer,
                padding=True,
                return_tensors="pt"
            )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Train
            print("Starting training...")
            self.trainer.train()
            
            # Save the model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            print(f"Training completed. Model saved to {output_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return False
    
    def evaluate(self, eval_dataset):
        """Evaluate the model"""
        if self.trainer is None:
            raise ValueError("Model has not been trained yet")
        
        try:
            eval_results = self.trainer.evaluate(eval_dataset)
            return eval_results
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return None
    
    def predict(self, texts: List[str]):
        """Make predictions on new texts"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        try:
            # Tokenize inputs
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.model.to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(predictions, dim=-1)
            
            return predicted_classes.cpu().numpy(), predictions.cpu().numpy()
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return None, None
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"Model saved to {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str):
        """Load a trained model"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Example data
    texts = [
        "This is a positive example",
        "This is a negative example",
        "Another positive text",
        "Another negative text"
    ]
    labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative
    
    # Initialize fine-tuner
    fine_tuner = FineTuner()
    
    # Prepare data
    train_dataset, eval_dataset = fine_tuner.prepare_data(texts, labels)
    
    # Train model
    success = fine_tuner.train(train_dataset, eval_dataset, num_epochs=1)
    
    if success:
        print("Training completed successfully!")
        
        # Make predictions
        test_texts = ["This is a great example", "This is terrible"]
        predictions, probabilities = fine_tuner.predict(test_texts)
        
        print("Predictions:", predictions)
        print("Probabilities:", probabilities)