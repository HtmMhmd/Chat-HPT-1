"""
Simplified inference implementation for the Chat2-124M project.
"""

import numpy as np
from tokenizer import SimpleTokenizer
from model import SimpleTransformer

class SimpleInference:
    """A minimal inference implementation."""
    
    def __init__(self, model, tokenizer):
        """Initialize the inference engine.
        
        Args:
            model: A SimpleTransformer instance
            tokenizer: A SimpleTokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_text(self, prompt, max_new_tokens=50, temperature=0.7):
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text
        """
        # Tokenize the prompt
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = np.array([prompt_ids])  # Add batch dimension
        
        # Generate from the model
        generated_ids = self.model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(generated_ids[0])
        
        # Extract only the newly generated part (after the prompt)
        response = generated_text[len(prompt):]
        
        return response
    
    def answer_question(self, context, question, max_new_tokens=50):
        """Answer a question based on context.
        
        Args:
            context: Context text
            question: Question to answer
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated answer
        """
        # Format as QA prompt
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        # Generate answer
        answer = self.generate_text(prompt, max_new_tokens=max_new_tokens)
        
        return answer.strip()
    
    def complete_form(self, context, fields):
        """Complete form fields based on context.
        
        Args:
            context: Context text
            fields: List of form fields to fill
            
        Returns:
            Dictionary mapping fields to values
        """
        results = {}
        
        for field in fields:
            # Format as a form filling prompt
            prompt = f"Context: {context}\nFill the form field: {field}\nValue:"
            
            # Generate field value
            value = self.generate_text(prompt, max_new_tokens=20)
            
            # Store the result
            results[field] = value.strip()
        
        return results

# Example usage
if __name__ == "__main__":
    # Create tokenizer and model
    tokenizer = SimpleTokenizer()
    model = SimpleTransformer(vocab_size=len(tokenizer.vocab), d_model=64, n_heads=2, n_layers=2)
    
    # Create inference engine
    inference = SimpleInference(model, tokenizer)
    
    # Example text generation
    prompt = "The future of AI is"
    print(f"Prompt: {prompt}")
    print(f"Completion: {inference.generate_text(prompt, max_new_tokens=20)}")
    
    # Example question answering
    context = "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity."
    question = "Where was Einstein born?"
    print(f"\nContext: {context}")
    print(f"Question: {question}")
    print(f"Answer: {inference.answer_question(context, question)}")
