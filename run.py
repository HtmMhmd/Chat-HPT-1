"""
Simple command-line interface for the Chat2-124M project.
"""

import argparse
import json

from tokenizer import SimpleTokenizer
from model import SimpleTransformer
from data import SimpleDataLoader
from inference import SimpleInference
from evaluate import SimpleEvaluator
from train import SimpleTrainer

def create_basic_model():
    """Create a basic tokenizer and model for demonstration."""
    tokenizer = SimpleTokenizer()
    model = SimpleTransformer(vocab_size=len(tokenizer.vocab), d_model=64, n_heads=2, n_layers=2)
    return tokenizer, model

def run_training(args):
    """Run model training."""
    print("Setting up training...")
    tokenizer, model = create_basic_model()
    trainer = SimpleTrainer(model, tokenizer, learning_rate=args.learning_rate)
    
    # Load training data
    data_loader = SimpleDataLoader()
    texts = []
    
    with open(args.input, 'r') as f:
        for line in f:
            texts.append(line.strip())
    
    print(f"Loaded {len(texts)} examples")
    
    # Prepare dataset
    dataset = data_loader.prepare_for_training(texts, tokenizer, max_length=args.max_length)
    
    # Train model
    trainer.train(dataset, batch_size=args.batch_size, num_epochs=args.epochs)
    
    # Save model (in a real implementation)
    trainer.save_model(args.output)

def run_inference(args):
    """Run model inference."""
    print("Setting up inference...")
    tokenizer, model = create_basic_model()
    inference = SimpleInference(model, tokenizer)
    
    # Generate from prompt
    prompt = args.prompt or input("Enter prompt: ")
    generated = inference.generate_text(prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
    
    print(f"\nPrompt: {prompt}")
    print(f"Completion: {generated}")

def run_evaluation(args):
    """Run model evaluation."""
    print("Setting up evaluation...")
    tokenizer, model = create_basic_model()
    evaluator = SimpleEvaluator(model, tokenizer)
    
    # Load test data
    with open(args.input, 'r') as f:
        test_data = json.load(f)
    
    metrics = {}
    
    # Calculate perplexity
    if 'texts' in test_data:
        perplexity = evaluator.calculate_perplexity(test_data['texts'])
        metrics['perplexity'] = perplexity
    
    # Evaluate QA
    if all(k in test_data for k in ['contexts', 'questions', 'answers']):
        qa_metrics = evaluator.evaluate_qa(
            test_data['contexts'], test_data['questions'], test_data['answers']
        )
        metrics.update(qa_metrics)
    
    # Evaluate form completion
    if all(k in test_data for k in ['form_contexts', 'form_fields', 'form_values']):
        form_metrics = evaluator.evaluate_form_completion(
            test_data['form_contexts'], test_data['form_fields'], test_data['form_values']
        )
        metrics.update({f"form_{k}": v for k, v in form_metrics.items()})
    
    # Log metrics
    evaluator.log_metrics(metrics)

def main():
    """Parse command-line arguments and run appropriate function."""
    parser = argparse.ArgumentParser(description="Simple Chat2-124M CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--input', required=True, help='Input file with training texts')
    train_parser.add_argument('--output', default='model.pkl', help='Where to save the model')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    
    # Inference arguments
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--prompt', help='Prompt for generation')
    infer_parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens to generate')
    infer_parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    
    # Evaluation arguments
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--input', required=True, help='Test data file (JSON)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_training(args)
    elif args.command == 'infer':
        run_inference(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
