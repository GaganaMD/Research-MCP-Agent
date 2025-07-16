# scripts/fine_tune_model.py
import argparse
import yaml
from pathlib import Path
import logging
from src.models.model_manager import ModelManager
from src.models.fine_tuner import LLMFineTuner

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for research tasks")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--training-data", type=str, required=True, help="Training data path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--base-model", type=str, help="Base model name")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    model_manager = ModelManager(config)
    fine_tuner = LLMFineTuner(model_manager)
    
    # Load base model
    base_model = args.base_model or config["base_model"]
    model_manager.load_base_model(base_model)
    
    # Prepare training data
    training_data = fine_tuner.prepare_training_data(args.training_data)
    
    # Fine-tune
    logging.info(f"Starting fine-tuning of {base_model}...")
    fine_tuner.fine_tune_model(base_model, training_data, args.output_dir)
    
    logging.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()