# Chat-HPT-1: Simple Transformer-based Chat Model

<img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9+-blue.svg">
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
<img alt="Docker" src="https://img.shields.io/badge/Docker-Supported-blue.svg">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Powered-red.svg">
<img alt="MLflow" src="https://img.shields.io/badge/MLflow-Tracking-green.svg">

A minimal implementation of a transformer-based language model for chatbots and text generation.

## Project Overview

Chat-HPT-1 is a lightweight transformer implementation designed for text generation and simple question answering. It includes:

- A minimal transformer model implementation
- Tokenizer with basic BPE support
- Training and inference pipelines
- Evaluation tools
- MLflow integration for experiment tracking

## Directory Structure

```
project_dir/
├── app/
│   ├── __init__.py
│   ├── inference.py          # Main inference logic
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py    # Transformer model implementation
│   └── utils/
│       ├── __init__.py
│       ├── data.py           # Data loading utilities
│       ├── evaluation.py     # Model evaluation utilities
│       └── tokenizer.py      # Text tokenization utilities
├── config/
│   └── config.yaml           # Configuration parameters
├── data/
│   ├── text/                 # Input text for training
│   │   └── training_data.txt
│   └── models/
│       └── model_checkpoint.pkl  # Trained model weights
├── output/
│   └── inference_outputs/    # Generated text outputs
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker services configuration
├── requirements.txt          # Python dependencies
├── run.py                    # CLI entry point
└── README.md                 # Project documentation
```

## API Usage Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│                 │     │                  │     │               │
│  Input Text     │────►│  Tokenizer       │────►│  Transformer  │
│                 │     │                  │     │  Encoder      │
└─────────────────┘     └──────────────────┘     └───────┬───────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│                 │     │                  │     │               │
│  Generated Text │◄────│  Detokenizer     │◄────│  Transformer  │
│                 │     │                  │     │  Decoder      │
└─────────────────┘     └────────┬─────────┘     └───────▲───────┘
                                 │                       │
                                 ▼                       │
                        ┌──────────────────┐     ┌───────────────┐
                        │                  │     │               │
                        │  Post-processing │     │  Attention    │
                        │                  │     │  Mechanism    │
                        └──────────────────┘     └───────────────┘
```

## Installation

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Chat-HPT-1.git
   cd Chat-HPT-1
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation

1. Build and run using Docker Compose:
   ```bash
   docker-compose up --build
   ```

## Usage

### Training a Model

```bash
# Train a new model
python run.py train --input data/text/training_data.txt

# Continue training from an existing model
python run.py train --input data/text/training_data.txt --model data/models/model_checkpoint.pkl --tokenizer data/models/tokenizer_checkpoint.pkl
```

### Running Inference

```bash
# Generate text from a prompt
python run.py infer --model data/models/model_checkpoint.pkl --tokenizer data/models/tokenizer_checkpoint.pkl --prompt "The future of AI is"

# Interactive mode
python run.py infer --model data/models/model_checkpoint.pkl --tokenizer data/models/tokenizer_checkpoint.pkl
```

### Evaluating a Model

```bash
# Evaluate on test data
python run.py evaluate --model data/models/model_checkpoint.pkl --tokenizer data/models/tokenizer_checkpoint.pkl --input data/text/test_data.json
```

## Experiment Tracking with MLflow

This project uses MLflow for experiment tracking and model management.

### Starting the MLflow UI

```bash
mlflow ui --port 5000
```

Then open your browser to http://localhost:5000 to view the MLflow UI.

### Tracked Metrics

- Training Loss
- Perplexity
- QA Accuracy
- Form Completion Accuracy

### Model Registry

Models can be registered with MLflow by setting `register_model: true` in the config.yaml file.

## Configuration

Model parameters, training settings, and inference configurations can be adjusted in the `config/config.yaml` file:

```yaml
# Model configuration
model:
  vocab_size: 256
  d_model: 64
  n_heads: 2
  n_layers: 2
  max_seq_length: 512

# Training configuration
training:
  batch_size: 4
  num_epochs: 3
  learning_rate: 0.001
  
# Inference configuration
inference:
  max_new_tokens: 50
  temperature: 0.7
```

## ⚙️ Configuration

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | Operation mode: `train`, `infer`, or `evaluate` |
| `--input` | Input file path for training/evaluation data |
| `--model` | Path to model checkpoint |
| `--tokenizer` | Path to tokenizer model |
| `--prompt` | Text prompt for inference |
| `--max-tokens` | Maximum tokens to generate |
| `--temperature` | Sampling temperature (higher = more random) |
| `--batch-size` | Batch size for training |
| `--epochs` | Number of training epochs |

## Docker Support

The project includes Docker support for reproducible environments:

```bash
# Build and run the application
docker-compose up

# Run a specific command
docker-compose run app train --input data/text/training_data.txt
```

## 🔍 Detailed System Explanation

The Chat-HPT-1 system implements a complete transformer-based architecture for text generation:

1. **Tokenization Layer**
   - Byte-Pair Encoding (BPE) for subword tokenization
   - Dictionary-based vocabulary with special tokens
   - Dynamic vocabulary expansion during training

2. **Transformer Layer**
   - Multi-head self-attention mechanism
   - Position-wise feed-forward networks
   - Layer normalization and residual connections
   - Positional encoding for sequence order information

3. **Training Layer**
   - Cross-entropy loss function for next-token prediction
   - Adam optimizer with learning rate scheduling
   - Gradient clipping to prevent exploding gradients
   - MLflow integration for experiment tracking

4. **Generation Layer**
   - Temperature-based sampling for controllable randomness
   - Beam search for higher quality generation
   - Length penalty and repetition penalties
   - Post-processing filters

### Key Components

#### Transformer Model

The core transformer architecture follows the "Attention Is All You Need" paper:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attention_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x
```

#### Tokenizer

The BPE tokenizer handles text encoding and decoding:

```python
def tokenize(self, text):
    """
    Convert text to token IDs using BPE algorithm.
    
    Args:
        text: Input text string
        
    Returns:
        List of token IDs
    """
    # Initialize with character-level splits
    pieces = list(text)
    
    # Merge pairs according to BPE merges
    while len(pieces) > 1:
        pairs = self._get_pairs(pieces)
        if not pairs:
            break
            
        # Find the highest-ranked pair
        best_pair = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
        if best_pair not in self.merges:
            break
            
        # Merge the pair
        pieces = self._merge_pair(best_pair[0], best_pair[1], pieces)
    
    # Convert pieces to token IDs
    return [self.token_to_id.get(p, self.token_to_id['<unk>']) for p in pieces]
```

#### Training Loop

The training pipeline handles data loading, model updates, and logging:

```python
def train_epoch(model, dataloader, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model: Transformer model
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids)
        loss = calculate_loss(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
