"""
Inference implementation for the Chat-HPT-1 project.
"""

import numpy as np
import mlflow
import os
import json

class SimpleInference:
    """A minimal inference implementation."""
    
    def __init__(self, model, tokenizer, config=None):
        """Initialize the inference engine.
        
        Args:
            model: A SimpleTransformer instance
            tokenizer: A SimpleTokenizer instance
            config: Optional configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
    
    def generate_text(self, prompt, max_new_tokens=50, temperature=0.7):
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text
        """
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param("prompt", prompt)
            mlflow.log_param("max_new_tokens", max_new_tokens)
            mlflow.log_param("temperature", temperature)
            
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
            
            # Log output
            mlflow.log_param("response", response)
            
            # Save to file if output_dir is configured
            output_dir = self.config.get("output_dir", "output/inference_outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            output = {
                "prompt": prompt,
                "response": response,
                "full_text": generated_text,
                "params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature
                }
            }
            
            filename = f"{output_dir}/generation_{run.info.run_id}.json"
            with open(filename, "w") as f:
                json.dump(output, f, indent=2)
            
            mlflow.log_artifact(filename)
            
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
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param("context_length", len(context))
            mlflow.log_param("question", question)
            
            # Format as QA prompt
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            
            # Generate answer
            answer = self.generate_text(prompt, max_new_tokens=max_new_tokens)
            
            mlflow.log_param("answer", answer)
            
            return answer.strip()
    
    def complete_form(self, context, fields):
        """Complete form fields based on context.
        
        Args:
            context: Context text
            fields: List of form fields to fill
            
        Returns:
            Dictionary mapping fields to values
        """
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param("context_length", len(context))
            mlflow.log_param("num_fields", len(fields))
            
            results = {}
            
            for field in fields:
                # Format as a form filling prompt
                prompt = f"Context: {context}\nFill the form field: {field}\nValue:"
                
                # Generate field value
                value = self.generate_text(prompt, max_new_tokens=20)
                
                # Store the result
                results[field] = value.strip()
            
            # Log results
            output_dir = self.config.get("output_dir", "output/inference_outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            output = {
                "context": context,
                "fields": fields,
                "results": results
            }
            
            filename = f"{output_dir}/form_completion_{run.info.run_id}.json"
            with open(filename, "w") as f:
                json.dump(output, f, indent=2)
            
            mlflow.log_artifact(filename)
            
            return results
