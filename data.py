"""
Simplified data loading and preprocessing for the Chat2-124M project.
"""

class SimpleDataLoader:
    """A minimal data loader implementation."""
    
    def __init__(self):
        """Initialize the data loader."""
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
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
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

# Example usage
if __name__ == "__main__":
    data_loader = SimpleDataLoader()
    
    # Example loading and cleaning
    text = "This  is  an   example   text  with  extra  spaces."
    cleaned_text = data_loader.clean_text(text)
    print(f"Original: '{text}'")
    print(f"Cleaned:  '{cleaned_text}'")
    
    # Example prompt creation
    context = "John Smith is 35 years old and works as a software engineer."
    question = "What is John's age?"
    qa_prompt = data_loader.create_qa_prompt(context, question)
    print(f"\nQA Prompt:\n{qa_prompt}")
