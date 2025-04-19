"""
Data loading and preprocessing utilities for the Chat-HPT-1 project.
"""
import os
import numpy as np
import mlflow

class SimpleDataLoader:
    """A minimal data loader implementation."""
    
    def __init__(self, config=None):
        """Initialize the data loader.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.loaded_texts = []
        self.cleaned_texts = []
        self.training_ready = False
    
    def load_text_file(self, file_path):
        """Load text from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Text content as a string
        """
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param("input_file", file_path)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            mlflow.log_metric("text_length", len(text))
            return text
    
    def load_pdf(self, file_path):
        """Extract text from a PDF file (simplified).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        # In a real implementation, we would use a library like PyPDF2 or pdfminer.six
        print(f"Simulating PDF extraction from {file_path}")
        return f"This is the extracted text from {file_path}."
    
    def clean_text(self, text):
        """Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def prepare_for_training(self, texts, tokenizer, max_length=512):
        """Prepare text data for training.
        
        Args:
            texts: List of text samples
            tokenizer: A tokenizer instance
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and labels
        """
        all_input_ids = []
        all_labels = []
        
        with mlflow.start_run(nested=True) as run:
            mlflow.log_param("max_length", max_length)
            mlflow.log_param("num_samples", len(texts))
            
            for text in texts:
                # Tokenize the text
                token_ids = tokenizer.encode(text)
                
                # Truncate if too long
                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                
                # Create labels (shifted input_ids)
                labels = token_ids[1:] + [tokenizer.special_tokens["<PAD>"]]
                
                all_input_ids.append(token_ids)
                all_labels.append(labels)
            
            # Log average sequence length
            avg_seq_len = np.mean([len(ids) for ids in all_input_ids])
            mlflow.log_metric("avg_sequence_length", avg_seq_len)
        
        return {
            "input_ids": all_input_ids,
            "labels": all_labels
        }
    
    def create_qa_prompt(self, context, question):
        """Create a prompt for question answering.
        
        Args:
            context: Context text
            question: Question text
            
        Returns:
            Formatted prompt
        """
        return f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    def create_form_prompt(self, context, field):
        """Create a prompt for form filling.
        
        Args:
            context: Context text
            field: Form field to fill
            
        Returns:
            Formatted prompt
        """
        return f"Context: {context}\nFill the form field: {field}\nValue:"
