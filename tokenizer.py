"""
Simplified tokenizer implementation for the Chat2-124M project.
"""
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
        """Learn BPE merge operations from texts."""
        # Count token pairs
        pairs = {}
        for text in texts:
            tokens = list(text)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair not in pairs:
                    pairs[pair] = 0
                pairs[pair] += 1
                
        # Add most frequent pairs as merges
        for _ in range(num_merges):
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            if not self.add_bpe_merge(*best_pair):
                del pairs[best_pair]
                continue
                
            # Update pair counts (simplified)
            new_pairs = {}
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

# Example usage
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()
    
    # Train BPE on some example texts
    tokenizer.learn_bpe(["hello world", "hello there", "world peace"], num_merges=5)
    
    # Test encoding and decoding
    text = "hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    
    print(f"Original: {text}")
    print(f"Encoded: {ids}")
    print(f"Decoded: {decoded}")
