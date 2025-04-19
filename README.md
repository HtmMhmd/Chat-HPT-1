# Chat-HPT-1: Simple Transformer-based Chat Model

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

## Docker Support

The project includes Docker support for reproducible environments:

```bash
# Build and run the application
docker-compose up

# Run a specific command
docker-compose run app train --input data/text/training_data.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
