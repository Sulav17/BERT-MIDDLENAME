# BERT-MIDDLENAME
MiddleNameNorm: BERT-based Middle Name Normalization from Full Names

## Overview

BERT-MIDDLENAME is a FastAPI service that predicts the middle name from a full name using a BERT-based classifier. It also includes a training pipeline to fine-tune the BERT model on custom data.

## Prerequisites
  
- Docker
- Python 3.9+ (if running without Docker)
- Git LFS (for downloading the base BERT model)

## Installation (without Docker)

1. Clone the repo:
   ```bash
   git clone <repo_url>
   cd BERT-MIDDLENAME
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Download the pretrained BERT model into `local-bert/`:
   ```bash
   # Initialize Git LFS if needed
   git lfs install
   # Clone the official BERT base uncased model (or any other BERT) locally
   git clone https://huggingface.co/bert-base-uncased local-bert
   ```
   Alternatively, you can use the Hugging Face CLI:
   ```bash
   huggingface-cli repo clone bert-base-uncased local-bert
   ```

## Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t bert-middle-name .
   ```
2. Run the Docker container (it starts the API service using the bundled model):
   ```bash
   docker run -d -p 8347:8347 --name bert-middle-name-app bert-middle-name
   ```
3. Access the API at `http://localhost:8347`.

## Configuration
  
Copy `.env.example` to `.env` and adjust values as needed:
```bash
# Port for the API server (if unset, defaults to 8347 in `entrypoint.sh`)
PORT=8347
# Path to the fine-tuned model directory (optional; defaults to ./models/bert-middle-name)
MODEL_PATH=./models/bert-middle-name
```

## API Endpoints

- `GET /`  
  Welcome message.
- `POST /predict-middle-name`  
  Predict the middle name from a full name.
  ```bash
  curl -X POST http://localhost:8347/predict-middle-name \
       -H 'Content-Type: application/json' \
       -d '{"full_name": "John Richard Doe"}'
  ```
> **Note:** Training is performed via the `app/trainer.py` script (see "Training the Model" section), not through an API endpoint.

## Training the Model

1. Ensure you have downloaded the pretrained BERT model into `local-bert/`.
2. Prepare your data:
   - `data/train.json`: training set with `full_name` and `middle_name` fields.
   - `data/valid.json`: validation set with the same format.
3. Run the training script:
   ```bash
   python app/trainer.py
   ```
   This will:
   - Load the base model and tokenizer from `local-bert/`.
   - Auto-detect GPU/CPU and train accordingly.
   - Save the fine-tuned model and tokenizer to `models/bert-middle-name`.
   - Write logs to the `logs/` directory.
4. (Optional) To force training on a specific GPU:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python app/trainer.py
   ```

## Project Structure
  
```
.                              # project root
├── app/                       # application code
│   ├── main.py                # FastAPI entrypoint & routes
│   ├── model.py               # model loading & inference
│   ├── schema.py              # Pydantic request/response schemas
│   ├── trainer.py             # training / fine-tuning script
│   └── utils/                 # helper modules
│       └── label_map.py       # builds label mappings from `data/`
├── data/                      # dataset files for training/validation
│   ├── train.json             # training set
│   └── valid.json             # validation set
├── local-bert/                # base BERT model for fine-tuning
│   ├── config.json
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── model.safetensors
├── models/                    # directory to save fine-tuned models
│   └── bert-middle-name       # fine-tuned model outputs
├── logs/                      # training logs
├── Dockerfile                 # Docker build specification
├── entrypoint.sh              # container entrypoint script
├── requirements.txt           # Python dependencies
├── .env.example               # environment variable template
└── README.md                  # this file
```

## CI/CD

A GitHub Actions workflow is included to build and test the Docker image:  
`.github/workflows/docker-ci.yml`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
