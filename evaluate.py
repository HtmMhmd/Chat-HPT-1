"""
Simplified evaluation implementation for the Chat2-124M project.
"""

import numpy as np
from tokenizer import SimpleTokenizer
from model import SimpleTransformer
from inference import SimpleInference

class SimpleEvaluator:
    """A minimal model evaluator implementation."""
    
    def __init__(self, model, tokenizer):
        """Initialize the evaluator.
        
        Args:
            model: A SimpleTransformer instance
            tokenizer: A SimpleTokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.inference = SimpleInference(model, tokenizer)
    
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
        correct = 0
        
        for i in range(len(questions)):
            context = contexts[i]
            question = questions[i]
            truth = ground_truth[i]
            
            # Get model prediction
            predicted = self.inference.answer_question(context, question)
            
            # Check exact match
            if predicted.lower() == truth.lower():
                correct += 1
                
        # Calculate accuracy
        accuracy = correct / len(questions) if len(questions) > 0 else 0
        
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
        correct_fields = 0
        total_fields = 0
        
        for i in range(len(contexts)):
            context = contexts[i]
            fields = field_lists[i]
            truth = ground_truths[i]
            
            # Get model predictions
            predictions = self.inference.complete_form(context, fields)
            
            # Check each field
            for field in fields:
                if field in truth and field in predictions:
                    if predictions[field].lower() == truth[field].lower():
                        correct_fields += 1
                total_fields += 1
        
        # Calculate accuracy
        accuracy = correct_fields / total_fields if total_fields > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct_fields": correct_fields,
            "total_fields": total_fields
        }
    
    def log_metrics(self, metrics):
        """Log evaluation metrics (simplified).
        
        Args:
            metrics: Dictionary of metrics to log
        """
        print("\nEvaluation Results:")
        for name, value in metrics.items():
            print(f"  {name}: {value}")

# Example usage
if __name__ == "__main__":
    # Create tokenizer and model
    tokenizer = SimpleTokenizer()
    model = SimpleTransformer(vocab_size=len(tokenizer.vocab), d_model=64, n_heads=2, n_layers=2)
    
    # Create evaluator
    evaluator = SimpleEvaluator(model, tokenizer)
    
    # Calculate perplexity
    texts = [
        "This is a simple example.",
        "The model is being evaluated."
    ]
    perplexity = evaluator.calculate_perplexity(texts)
    print(f"Perplexity: {perplexity:.2f}")
    
    # Evaluate QA
    contexts = ["Albert Einstein was born in Germany.", "Python is a programming language."]
    questions = ["Where was Einstein born?", "What is Python?"]
    answers = ["Germany", "a programming language"]
    
    qa_metrics = evaluator.evaluate_qa(contexts, questions, answers)
    evaluator.log_metrics(qa_metrics)
