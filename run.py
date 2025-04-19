"""
Command-line interface for the Chat-HPT-1 project.
"""

import argparse
import os
import yaml
import json
import mlflow
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chat-hpt-1")

# Import local modules
from app.models import SimpleTransformer
from app.utils import SimpleTokenizer, SimpleDataLoader, SimpleEvaluator
from app.inference import SimpleInference

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_mlflow(config):
    """Set up MLflow tracking."""
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "./mlruns")
    experiment_name = mlflow_config.get("experiment_name", "chat-hpt-1")
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    
    return experiment

def train_model(args, config):
    """Train a model from scratch or continue training an existing model."""
    logger.info("Starting model training...")
    
    # Set up tracking
    experiment = setup_mlflow(config)
    
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        # Log configuration
        mlflow.log_params({
            "batch_size": args.batch_size or config["training"]["batch_size"],
            "num_epochs": args.epochs or config["training"]["num_epochs"],
            "learning_rate": args.learning_rate or config["training"]["learning_rate"],
            "input_file": args.input or config["paths"]["train_data"]
        })
        
        # Create or load tokenizer
        if args.tokenizer and os.path.exists(args.tokenizer):
            logger.info(f"Loading tokenizer from {args.tokenizer}")
            tokenizer = SimpleTokenizer.load(args.tokenizer)
        else:
            logger.info("Creating new tokenizer")
            tokenizer = SimpleTokenizer()
        
        # Create or load model
        if args.model and os.path.exists(args.model):
            logger.info(f"Loading model from {args.model}")
            model = SimpleTransformer.load(args.model)
        else:
            logger.info("Creating new model")
            model_config = config["model"]
            model = SimpleTransformer(
                vocab_size=len(tokenizer.vocab),
                d_model=model_config["d_model"],
                n_heads=model_config["n_heads"],
                n_layers=model_config["n_layers"]
            )
        
        # Set up training
        from app.utils.training import SimpleTrainer
        trainer = SimpleTrainer(
            model=model,
            tokenizer=tokenizer,
            learning_rate=args.learning_rate or config["training"]["learning_rate"],
            config=config
        )
        
        # Load training data
        data_loader = SimpleDataLoader(config)
        input_file = args.input or config["paths"]["train_data"]
        logger.info(f"Loading training data from {input_file}")
        
        text = data_loader.load_text_file(input_file)
        texts = [text[i:i+1000] for i in range(0, len(text), 1000)]  # Split into chunks
        
        logger.info(f"Preparing {len(texts)} text chunks for training")
        dataset = data_loader.prepare_for_training(
            texts=texts,
            tokenizer=tokenizer,
            max_length=config["model"]["max_seq_length"]
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train(
            dataset=dataset,
            batch_size=args.batch_size or config["training"]["batch_size"],
            num_epochs=args.epochs or config["training"]["num_epochs"]
        )
        
        # Save model and tokenizer
        model_dir = config["paths"]["model_dir"]
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"model_{run.info.run_id}.pkl")
        tokenizer_path = os.path.join(model_dir, f"tokenizer_{run.info.run_id}.pkl")
        
        logger.info(f"Saving model to {model_path}")
        model.save(model_path)
        
        logger.info(f"Saving tokenizer to {tokenizer_path}")
        tokenizer.save(tokenizer_path)
        
        # Log artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(tokenizer_path)
        
        # Register model if configured
        if config["mlflow"].get("register_model", False):
            mlflow.register_model(
                f"runs:/{run.info.run_id}/{model_path}",
                "chat-hpt-1"
            )
        
        logger.info("Training completed successfully")

def run_inference(args, config):
    """Run inference using a trained model."""
    logger.info("Starting inference...")
    
    # Set up tracking
    experiment = setup_mlflow(config)
    
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        # Load tokenizer and model
        if not args.model or not os.path.exists(args.model):
            logger.error("Model path is required for inference")
            return
        
        if not args.tokenizer or not os.path.exists(args.tokenizer):
            logger.error("Tokenizer path is required for inference")
            return
        
        logger.info(f"Loading model from {args.model}")
        model = SimpleTransformer.load(args.model)
        
        logger.info(f"Loading tokenizer from {args.tokenizer}")
        tokenizer = SimpleTokenizer.load(args.tokenizer)
        
        # Set up inference
        inference = SimpleInference(model=model, tokenizer=tokenizer, config=config)
        
        # Log parameters
        mlflow.log_params({
            "model_path": args.model,
            "tokenizer_path": args.tokenizer,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens or config["inference"]["max_new_tokens"],
            "temperature": args.temperature or config["inference"]["temperature"]
        })
        
        # Run inference
        prompt = args.prompt or input("Enter prompt: ")
        max_tokens = args.max_tokens or config["inference"]["max_new_tokens"]
        temperature = args.temperature or config["inference"]["temperature"]
        
        logger.info(f"Generating text for prompt: {prompt}")
        response = inference.generate_text(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        # Display results
        print("\n=== Generated Text ===")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("======================")
        
        # Log results
        mlflow.log_param("response", response)
        
        logger.info("Inference completed successfully")

def evaluate_model(args, config):
    """Evaluate a trained model on test data."""
    logger.info("Starting model evaluation...")
    
    # Set up tracking
    experiment = setup_mlflow(config)
    
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"evaluate-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        # Load tokenizer and model
        if not args.model or not os.path.exists(args.model):
            logger.error("Model path is required for evaluation")
            return
        
        if not args.tokenizer or not os.path.exists(args.tokenizer):
            logger.error("Tokenizer path is required for evaluation")
            return
        
        if not args.input or not os.path.exists(args.input):
            logger.error("Test data path is required for evaluation")
            return
        
        logger.info(f"Loading model from {args.model}")
        model = SimpleTransformer.load(args.model)
        
        logger.info(f"Loading tokenizer from {args.tokenizer}")
        tokenizer = SimpleTokenizer.load(args.tokenizer)
        
        # Log parameters
        mlflow.log_params({
            "model_path": args.model,
            "tokenizer_path": args.tokenizer,
            "test_data": args.input
        })
        
        # Load test data
        logger.info(f"Loading test data from {args.input}")
        with open(args.input, 'r') as f:
            test_data = json.load(f)
        
        # Set up evaluator
        evaluator = SimpleEvaluator(model=model, tokenizer=tokenizer, config=config)
        
        # Run evaluations
        metrics = {}
        
        # Calculate perplexity if test texts are available
        if 'texts' in test_data:
            logger.info(f"Calculating perplexity on {len(test_data['texts'])} texts")
            perplexity = evaluator.calculate_perplexity(test_data['texts'])
            metrics['perplexity'] = perplexity
        
        # Evaluate QA if applicable
        if all(k in test_data for k in ['contexts', 'questions', 'answers']):
            logger.info(f"Evaluating QA on {len(test_data['questions'])} examples")
            qa_metrics = evaluator.evaluate_qa(
                test_data['contexts'],
                test_data['questions'],
                test_data['answers']
            )
            metrics.update(qa_metrics)
        
        # Evaluate form completion if applicable
        if all(k in test_data for k in ['form_contexts', 'form_fields', 'form_values']):
            logger.info(f"Evaluating form completion on {len(test_data['form_contexts'])} examples")
            form_metrics = evaluator.evaluate_form_completion(
                test_data['form_contexts'],
                test_data['form_fields'],
                test_data['form_values']
            )
            metrics.update({f"form_{k}": v for k, v in form_metrics.items()})
        
        # Log metrics
        evaluator.log_metrics(metrics)
        
        logger.info("Evaluation completed successfully")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Chat-HPT-1 Command Line Interface")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training subparser
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--input", help="Path to training data")
    train_parser.add_argument("--model", help="Path to save/load model")
    train_parser.add_argument("--tokenizer", help="Path to save/load tokenizer")
    train_parser.add_argument("--batch-size", type=int, help="Training batch size")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    
    # Inference subparser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model", required=True, help="Path to model")
    infer_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer")
    infer_parser.add_argument("--prompt", help="Text prompt")
    infer_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    infer_parser.add_argument("--temperature", type=float, help="Sampling temperature")
    
    # Evaluation subparser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--model", required=True, help="Path to model")
    eval_parser.add_argument("--tokenizer", required=True, help="Path to tokenizer")
    eval_parser.add_argument("--input", required=True, help="Path to test data (JSON)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run the appropriate command
    if args.command == "train":
        train_model(args, config)
    elif args.command == "infer":
        run_inference(args, config)
    elif args.command == "evaluate":
        evaluate_model(args, config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
