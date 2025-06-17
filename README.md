# BERT-MIDDLENAME
MiddleNameNorm: BERT-based Middle Name Normalization from Full Names

## Overview

BERT-MIDDLENAME is a FastAPI service that predicts the middle name from a full name using a BERT-based classifier. It also includes a training pipeline to fine-tune the BERT model on custom data.

## Prerequisites

- Docker
- Python 3.9+ (if running without Docker)

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

## Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t bert-middle-name .
   ```
2. Run the Docker container (it will train the model on startup and then start the API):
   ```bash
   docker run -d -p 8000:8000 --name bert-middle-name-app bert-middle-name
   ```
3. Access the API at `http://localhost:8000`.

## Environment Variables

Copy `.env.example` to `.env` and adjust values as needed:
```bash
PORT=8000
MODEL_PATH=./models/bert-middle-name  # optional override of model directory
```

## API Endpoints

- `GET /`  
  Welcome message.
- `POST /predict-middle-name`  
  Predict the middle name from a full name.
  ```bash
  curl -X POST http://localhost:8000/predict-middle-name \
       -H 'Content-Type: application/json' \
       -d '{"full_name": "John Richard Doe"}'
  ```
- `POST /train`  
  Trigger model training manually.

## Project Structure

```
.  
├── app  
│   ├── main.py  
│   ├── model.py  
│   ├── schema.py  
│   ├── trainer.py  
│   └── utils  
│       └── label_map.py  
├── data  
│   ├── train.json  
│   └── valid.json  
├── logs  
├── models  
├── requirements.txt  
├── Dockerfile  
├── entrypoint.sh  
├── .dockerignore  
├── .gitignore  
├── .env.example  
└── README.md  
```

## CI/CD

A GitHub Actions workflow is included to build and test the Docker image:  
`.github/workflows/docker-ci.yml`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
