"""
Simplified transformer model implementation for the Chat2-124M project.
"""

import numpy as np

class SimpleTransformer:
    """A minimal transformer model implementation."""
    
    def __init__(self, vocab_size=256, d_model=64, n_heads=2, n_layers=2):
        """Initialize the transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model embeddings
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Initialize weights
        self.token_embedding = np.random.normal(0, 0.02, (vocab_size, d_model))
        self.positional_embedding = np.random.normal(0, 0.02, (1000, d_model))  # Max 1000 positions
        
        # Initialize layers (in a real implementation, these would be proper neural network layers)
        self.attention_weights = []
        self.ffn_weights = []
        for _ in range(n_layers):
            # Simple attention weights: query, key, value projections and output projection
            self.attention_weights.append({
                'q': np.random.normal(0, 0.02, (d_model, d_model)),
                'k': np.random.normal(0, 0.02, (d_model, d_model)),
                'v': np.random.normal(0, 0.02, (d_model, d_model)),
                'o': np.random.normal(0, 0.02, (d_model, d_model))
            })
            
            # Simple feed-forward network weights
            self.ffn_weights.append({
                'w1': np.random.normal(0, 0.02, (d_model, d_model * 4)),
                'b1': np.zeros(d_model * 4),
                'w2': np.random.normal(0, 0.02, (d_model * 4, d_model)),
                'b2': np.zeros(d_model)
            })
        
        # Output projection
        self.output_weights = np.random.normal(0, 0.02, (d_model, vocab_size))
        self.output_bias = np.zeros(vocab_size)
    
    def attention(self, q, k, v, mask=None):
        """Compute self-attention.
        
        Args:
            q: Query vectors [batch_size, seq_len, d_model]
            k: Key vectors [batch_size, seq_len, d_model]
            v: Value vectors [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        # Compute attention scores
        scores = np.matmul(q, np.transpose(k, [0, 2, 1]))
        scores = scores / np.sqrt(self.d_model)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)
            
        # Apply softmax
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-9)
        
        # Apply attention weights to values
        output = np.matmul(weights, v)
        
        return output
    
    def feed_forward(self, x, weights):
        """Apply feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            weights: Dictionary of feed-forward weights
            
        Returns:
            Feed-forward output [batch_size, seq_len, d_model]
        """
        # First dense layer with GELU activation
        hidden = np.matmul(x, weights['w1']) + weights['b1']
        hidden = 0.5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + 0.044715 * np.power(hidden, 3))))
        
        # Second dense layer
        output = np.matmul(hidden, weights['w2']) + weights['b2']
        
        return output
    
    def forward(self, input_ids):
        """Forward pass through the transformer.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert token IDs to embeddings
        embedding = np.zeros((batch_size, seq_len, self.d_model))
        for i in range(batch_size):
            for j in range(seq_len):
                token_id = input_ids[i, j]
                embedding[i, j] = self.token_embedding[token_id]
                embedding[i, j] += self.positional_embedding[j][:self.d_model]
        
        # Create causal mask (lower triangular)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask.reshape(1, seq_len, seq_len)
        
        # Process through transformer layers
        hidden_states = embedding
        for layer in range(self.n_layers):
            # Self-attention
            q = np.matmul(hidden_states, self.attention_weights[layer]['q'])
            k = np.matmul(hidden_states, self.attention_weights[layer]['k'])
            v = np.matmul(hidden_states, self.attention_weights[layer]['v'])
            
            attention_output = self.attention(q, k, v, mask)
            attention_output = np.matmul(attention_output, self.attention_weights[layer]['o'])
            
            # Residual connection and layer norm (simplified)
            hidden_states = hidden_states + attention_output
            hidden_states = hidden_states - np.mean(hidden_states, axis=-1, keepdims=True)
            hidden_states = hidden_states / (np.std(hidden_states, axis=-1, keepdims=True) + 1e-5)
            
            # Feed-forward network
            ffn_output = self.feed_forward(hidden_states, self.ffn_weights[layer])
            
            # Residual connection and layer norm (simplified)
            hidden_states = hidden_states + ffn_output
            hidden_states = hidden_states - np.mean(hidden_states, axis=-1, keepdims=True)
            hidden_states = hidden_states / (np.std(hidden_states, axis=-1, keepdims=True) + 1e-5)
        
        # Output projection
        logits = np.matmul(hidden_states, self.output_weights) + self.output_bias
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """Generate text by sampling from the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(input_ids)
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Convert to probabilities with softmax
            probs = np.exp(next_token_logits - np.max(next_token_logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            # Sample from the distribution
            next_token = np.zeros((input_ids.shape[0], 1), dtype=np.int32)
            for i in range(input_ids.shape[0]):
                next_token[i, 0] = np.random.choice(self.vocab_size, p=probs[i])
            
            # Concatenate with input_ids
            input_ids = np.concatenate([input_ids, next_token], axis=1)
        
        return input_ids

# Example usage
if __name__ == "__main__":
    # Create a simple transformer
    model = SimpleTransformer(vocab_size=256, d_model=64, n_heads=2, n_layers=2)
    
    # Generate from a prompt
    prompt_ids = np.array([[1, 20, 30, 40, 50]])  # Example input IDs
    generated_ids = model.generate(prompt_ids, max_new_tokens=10, temperature=0.7)
    
    print(f"Input shape: {prompt_ids.shape}")
    print(f"Output shape: {generated_ids.shape}")
    print(f"Generated IDs: {generated_ids[0].tolist()}")
