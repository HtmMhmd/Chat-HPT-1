"""
Training utilities for the Chat-HPT-1 project.
"""

import numpy as np
import time
import mlflow

class SimpleTrainer:
    """A minimal training implementation."""
    
    def __init__(self, model, tokenizer, learning_rate=0.001, config=None):
        """Initialize the trainer.
        
        Args:
            model: A SimpleTransformer instance
            tokenizer: A SimpleTokenizer instance
            learning_rate: Learning rate for optimization
            config: Optional configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.config = config or {}
        
        # Loss history
        self.losses = []
    
    def compute_loss(self, logits, labels):
        """Compute cross-entropy loss.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            
        Returns:
            Loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape logits and compute softmax
        logits_flat = logits.reshape(-1, vocab_size)
        log_probs = logits_flat - np.log(np.sum(np.exp(logits_flat), axis=-1, keepdims=True))
        
        # Gather log probabilities of target tokens
        labels_flat = labels.reshape(-1)
        target_log_probs = np.array([log_probs[i, labels_flat[i]] for i in range(len(labels_flat))])
        
        # Average loss
        loss = -np.mean(target_log_probs)
        
        return loss
    
    def train_step(self, batch):
        """Perform a single training step.
        
        Args:
            batch: Dictionary with input_ids and labels
            
        Returns:
            Loss value
        """
        # Forward pass
        logits = self.model.forward(batch["input_ids"])
        
        # Compute loss
        loss = self.compute_loss(logits, batch["labels"])
        
        # In a real implementation, we would compute gradients and update weights
        # Here we'll just simulate the weight update
        return loss
    
    def train(self, dataset, batch_size=4, num_epochs=3):
        """Train the model on a dataset.
        
        Args:
            dataset: Dictionary with input_ids and labels
            batch_size: Batch size
            num_epochs: Number of training epochs
        """
        num_samples = len(dataset["input_ids"])
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with mlflow.start_run(nested=True) as run:
            mlflow.log_params({
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": self.learning_rate,
                "num_samples": num_samples,
                "num_batches": num_batches,
            })
            
            print(f"Starting training with {num_samples} samples, {num_batches} batches per epoch")
            
            for epoch in range(num_epochs):
                # Shuffle dataset
                indices = np.random.permutation(num_samples)
                shuffled_input_ids = [dataset["input_ids"][i] for i in indices]
                shuffled_labels = [dataset["labels"][i] for i in indices]
                
                epoch_losses = []
                start_time = time.time()
                
                for batch_idx in range(num_batches):
                    # Get batch indices
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_size_actual = end_idx - start_idx
                    
                    # Create batch
                    batch_input_ids = shuffled_input_ids[start_idx:end_idx]
                    batch_labels = shuffled_labels[start_idx:end_idx]
                    
                    # Pad sequences to the same length within the batch
                    max_seq_len = max(len(ids) for ids in batch_input_ids)
                    padded_input_ids = np.zeros((batch_size_actual, max_seq_len), dtype=np.int32)
                    padded_labels = np.zeros((batch_size_actual, max_seq_len), dtype=np.int32)
                    
                    for i, (ids, labs) in enumerate(zip(batch_input_ids, batch_labels)):
                        padded_input_ids[i, :len(ids)] = ids
                        padded_labels[i, :len(labs)] = labs
                    
                    # Train on batch
                    batch = {
                        "input_ids": padded_input_ids,
                        "labels": padded_labels
                    }
                    
                    loss = self.train_step(batch)
                    epoch_losses.append(loss)
                    
                    # Print progress
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {loss:.4f}")
                
                # End of epoch
                epoch_loss = np.mean(epoch_losses)
                self.losses.append(epoch_loss)
                
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - Avg loss: {epoch_loss:.4f}")
                
                # Log metrics to MLflow
                mlflow.log_metric("loss", epoch_loss, step=epoch)
                mlflow.log_metric("epoch_time", epoch_time, step=epoch)
                
            # Final metrics
            final_loss = self.losses[-1]
            mlflow.log_metric("final_loss", final_loss)
            
            return final_loss
            
    def save_model(self, path):
        """Save the model.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        mlflow.log_artifact(path)
