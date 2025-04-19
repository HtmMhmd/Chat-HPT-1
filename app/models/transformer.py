"""
Simplified transformer model implementation for the Chat-HPT-1 project.
"""

import numpy as np
import mlflow

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
        """Compute self-attention using einsum for efficiency.
        
        Args:
            q, k, v: Query, Key, Value tensors [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        # Compute attention scores using einsum: batch, query_seq, key_seq
        scores = np.einsum('bij,bkj->bik', q, k)
        scores = scores / np.sqrt(self.d_model)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)
            
        # Apply softmax
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-9)
        
        # Apply attention weights to values using einsum
        output = np.einsum('bij,bjk->bik', weights, v)
        
        return output
    
    def feed_forward(self, x, weights):
        """Apply feed-forward network with einsum for efficiency.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            weights: Dictionary of feed-forward weights
            
        Returns:
            Feed-forward output [batch_size, seq_len, d_model]
        """
        # First dense layer with GELU activation
        hidden = np.einsum('bsi,ij->bsj', x, weights['w1']) + weights['b1']
        # GELU activation approximation
        hidden = 0.5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + 0.044715 * np.power(hidden, 3))))
        
        # Second dense layer
        output = np.einsum('bsi,ij->bsj', hidden, weights['w2']) + weights['b2']
        
        return output
    
    def forward(self, input_ids):
        """Forward pass through the transformer.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert token IDs to embeddings - vectorized approach
        embedding = self.token_embedding[input_ids]  # [batch_size, seq_len, d_model]
        embedding += self.positional_embedding[:seq_len]  # Broadcasting adds positional embeddings
        
        # Create causal mask (lower triangular)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask.reshape(1, seq_len, seq_len)
        
        # Process through transformer layers
        hidden_states = embedding
        for layer in range(self.n_layers):
            # Self-attention
            q = np.einsum('bsi,ij->bsj', hidden_states, self.attention_weights[layer]['q'])
            k = np.einsum('bsi,ij->bsj', hidden_states, self.attention_weights[layer]['k'])
            v = np.einsum('bsi,ij->bsj', hidden_states, self.attention_weights[layer]['v'])
            
            attention_output = self.attention(q, k, v, mask)
            attention_output = np.einsum('bsi,ij->bsj', attention_output, self.attention_weights[layer]['o'])
            
            # Residual connection and layer norm
            hidden_states = hidden_states + attention_output
            hidden_states = self.layer_norm(hidden_states)
            
            # Feed-forward network
            ffn_output = self.feed_forward(hidden_states, self.ffn_weights[layer])
            
            # Residual connection and layer norm
            hidden_states = hidden_states + ffn_output
            hidden_states = self.layer_norm(hidden_states)
        
        # Output projection
        logits = np.einsum('bsi,ij->bsj', hidden_states, self.output_weights) + self.output_bias
        
        return logits
    
    def layer_norm(self, x, epsilon=1e-5):
        """Apply layer normalization.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            epsilon: Small constant for numerical stability
            
        Returns:
            Normalized tensor [batch_size, seq_len, d_model]
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + epsilon)
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """Generate text by sampling from the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        batch_size = input_ids.shape[0]
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(input_ids)
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Convert to probabilities with softmax
            probs = self.softmax(next_token_logits)
            
            # Sample from the distribution
            next_token = np.zeros((batch_size, 1), dtype=np.int32)
            for i in range(batch_size):
                next_token[i, 0] = np.random.choice(self.vocab_size, p=probs[i])
            
            # Concatenate with input_ids
            input_ids = np.concatenate([input_ids, next_token], axis=1)
        
        return input_ids
    
    def softmax(self, x):
        """Apply softmax function.
        
        Args:
            x: Input tensor
            
        Returns:
            Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def save(self, filepath):
        """Save model weights to disk and log to MLflow.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model weights
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'token_embedding': self.token_embedding,
                'positional_embedding': self.positional_embedding,
                'attention_weights': self.attention_weights,
                'ffn_weights': self.ffn_weights,
                'output_weights': self.output_weights,
                'output_bias': self.output_bias
            }, f)
        
        # Log model to MLflow
        mlflow.log_artifact(filepath)
        mlflow.log_param("vocab_size", self.vocab_size)
        mlflow.log_param("d_model", self.d_model)
        mlflow.log_param("n_heads", self.n_heads)
        mlflow.log_param("n_layers", self.n_layers)
        
        print(f"Model saved to {filepath} and logged to MLflow")
    
    @classmethod
    def load(cls, filepath):
        """Load model weights from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded SimpleTransformer instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        
        # Create new model with the same architecture
        model = cls(
            vocab_size=weights['vocab_size'],
            d_model=weights['d_model'],
            n_heads=weights['n_heads'],
            n_layers=weights['n_layers']
        )
        
        # Load weights
        model.token_embedding = weights['token_embedding']
        model.positional_embedding = weights['positional_embedding']
        model.attention_weights = weights['attention_weights']
        model.ffn_weights = weights['ffn_weights']
        model.output_weights = weights['output_weights']
        model.output_bias = weights['output_bias']
        
        return model
