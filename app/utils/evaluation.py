"""
Evaluation utilities for the Chat-HPT-1 project.
"""

import numpy as np
import mlflow
import os
import json
from app.inference import SimpleInference

class SimpleEvaluator:
    """A minimal model evaluator implementation."""
    
    def __init__(self, model, tokenizer, config=None):
        """Initialize the evaluator.
        
        Args:
            model: A SimpleTransformer instance
            tokenizer: A SimpleTokenizer instance
            config: Optional configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        self.inference = SimpleInference(model, tokenizer, config)
    
    def calculate_perplexity(self, texts):
        """Calculate perplexity on a set of texts.
        
        Args:
            texts: List of text samples
            
        Returns:
            Perplexity score (lower is better)
        """
        total_log_likelihood = 0
        total_tokens = 0
        
        for text in texts:
            # Tokenize text
            token_ids = self.tokenizer.encode(text)
            
            # Create input and target
            input_ids = np.array([token_ids[:-1]])  # Remove last token for input
            target_ids = np.array([token_ids[1:]])   # Remove first token for target
            
            # Get model predictions
            logits = self.model.forward(input_ids)
            
            # Calculate log likelihood
            for i in range(logits.shape[1]):
                token_id = target_ids[0, i]
                token_logits = logits[0, i]
                log_softmax = token_logits - np.log(np.sum(np.exp(token_logits)))
                token_log_prob = log_softmax[token_id]
                total_log_likelihood += token_log_prob
                total_tokens += 1
        
        # Calculate perplexity
        perplexity = np.exp(-total_log_likelihood / total_tokens)
        
        return perplexity
    
    def evaluate_qa(self, contexts, questions, ground_truth):
        """Evaluate question answering performance.
        
        Args:
            contexts: List of context texts
            questions: List of questions
            ground_truth: List of ground truth answers
            
        Returns:
            Dictionary with accuracy and other metrics
        """
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param("num_qa_examples", len(questions))
            
            correct = 0
            predictions = []
            
            for i in range(len(questions)):
                context = contexts[i]
                question = questions[i]
                truth = ground_truth[i]
                
                # Get model prediction
                predicted = self.inference.answer_question(context, question)
                predictions.append(predicted)
                
                # Check exact match
                if predicted.lower() == truth.lower():
                    correct += 1
                    
            # Calculate accuracy
            accuracy = correct / len(questions) if len(questions) > 0 else 0
            
            # Log metrics
            mlflow.log_metric("qa_accuracy", accuracy)
            
            # Save predictions
            results = {
                "contexts": contexts,
                "questions": questions,
                "ground_truth": ground_truth,
                "predictions": predictions
            }
            
            output_dir = self.config.get("output_dir", "output/inference_outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            with open(f"{output_dir}/qa_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            mlflow.log_artifact(f"{output_dir}/qa_results.json")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(questions)
        }
    
    def evaluate_form_completion(self, contexts, field_lists, ground_truths):
        """Evaluate form completion performance.
        
        Args:
            contexts: List of context texts
            field_lists: List of lists of form fields
            ground_truths: List of ground truth values (dicts mapping fields to values)
            
        Returns:
            Dictionary with accuracy and other metrics
        """
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param("num_form_examples", len(contexts))
            
            correct_fields = 0
            total_fields = 0
            all_predictions = []
            
            for i in range(len(contexts)):
                context = contexts[i]
                fields = field_lists[i]
                truth = ground_truths[i]
                
                # Get model predictions
                predictions = self.inference.complete_form(context, fields)
                all_predictions.append(predictions)
                
                # Check each field
                for field in fields:
                    if field in truth and field in predictions:
                        if predictions[field].lower() == truth[field].lower():
                            correct_fields += 1
                    total_fields += 1
            
            # Calculate accuracy
            accuracy = correct_fields / total_fields if total_fields > 0 else 0
            
            # Log metrics
            mlflow.log_metric("form_accuracy", accuracy)
            
            # Save predictions
            results = {
                "contexts": contexts,
                "field_lists": field_lists,
                "ground_truths": ground_truths,
                "predictions": all_predictions
            }
            
            output_dir = self.config.get("output_dir", "output/inference_outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            with open(f"{output_dir}/form_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            mlflow.log_artifact(f"{output_dir}/form_results.json")
        
        return {
            "accuracy": accuracy,
            "correct_fields": correct_fields,
            "total_fields": total_fields
        }
    
    def log_metrics(self, metrics):
        """Log evaluation metrics to console and MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        print("\n=== Evaluation Results ===")
        for name, value in metrics.items():
            print(f"  {name}: {value}")
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)
        print("===========================")
