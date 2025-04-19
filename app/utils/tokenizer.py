"""
Tokenizer implementation for the Chat-HPT-1 project.
"""
import numpy as np
import os
import pickle

class SimpleTokenizer:
    """A minimal byte-level tokenizer with basic BPE functionality."""
    
    def __init__(self):
        """Initialize the simple tokenizer with basic vocabulary."""
        # Special tokens
        self.special_tokens = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        
        # Basic byte vocabulary (ASCII printable characters)
        self.vocab = {**self.special_tokens}
        for i in range(32, 127):  # ASCII printable characters
            self.vocab[chr(i)] = len(self.vocab)
            
        # Create a reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # BPE merges (in a real implementation, these would be learned)
        self.merges = {}  # Maps (token1, token2) -> merged_token
        
    def add_bpe_merge(self, token1, token2):
        """Add a BPE merge rule."""
        if token1 not in self.vocab or token2 not in self.vocab:
            return False
            
        merged = token1 + token2
        if merged in self.vocab:
            return False
            
        self.merges[(token1, token2)] = merged
        self.vocab[merged] = len(self.vocab)
        self.id_to_token[self.vocab[merged]] = merged
        return True

    def learn_bpe(self, texts, num_merges=10):
        """Learn BPE merge operations from texts using efficient counter."""
        # Count token pairs
        from collections import Counter
        
        pairs = Counter()
        for text in texts:
            tokens = list(text)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pairs[pair] += 1
                
        # Add most frequent pairs as merges
        for _ in range(num_merges):
            if not pairs:
                break
                
            best_pair = pairs.most_common(1)[0][0]
            if not self.add_bpe_merge(*best_pair):
                del pairs[best_pair]
                continue
                
            # Update pair counts (simplified)
            new_pairs = Counter()
            for pair, count in pairs.items():
                if pair != best_pair:
                    new_pairs[pair] = count
            pairs = new_pairs
    
    def encode(self, text):
        """Convert text to a list of token IDs."""
        ids = [self.special_tokens["<BOS>"]]
        
        # Simple character tokenization (in a real implementation, we'd apply BPE)
        for char in text:
            if char in self.vocab:
                ids.append(self.vocab[char])
            else:
                ids.append(self.special_tokens["<UNK>"])
                
        ids.append(self.special_tokens["<EOS>"])
        return ids
        
    def decode(self, ids):
        """Convert token IDs back to text."""
        text = ""
        for id in ids:
            if id in [self.special_tokens["<PAD>"], self.special_tokens["<BOS>"], 
                     self.special_tokens["<EOS>"], self.special_tokens["<UNK>"]]:
                continue
            text += self.id_to_token.get(id, "")
        return text
    
    def save(self, filepath):
        """Save tokenizer to disk.
        
        Args:
            filepath: Path to save the tokenizer
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'id_to_token': self.id_to_token,
                'merges': self.merges,
                'special_tokens': self.special_tokens
            }, f)
        
        print(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load tokenizer from disk.
        
        Args:
            filepath: Path to load the tokenizer from
            
        Returns:
            Loaded SimpleTokenizer instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls()
        tokenizer.vocab = data['vocab']
        tokenizer.id_to_token = data['id_to_token']
        tokenizer.merges = data['merges']
        tokenizer.special_tokens = data['special_tokens']
        
        return tokenizer
