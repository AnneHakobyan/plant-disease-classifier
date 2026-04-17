# Plant Disease Classifier

An AI system that identifies plant diseases from leaf images, regardless of the host plant species.

## Model
- Architecture: EfficientNet-B3 (timm)
- Classes: 82 disease categories
- Input: RGB leaf image
- Output: Disease name only (e.g. "late blight", not "tomato late blight")

## Setup

### 1. Clone the repo
git clone https://github.com/AnneHakobyan/plant-disease-classifier.git
cd plant-disease-classifier

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download model weights
Download best_model.pt from HuggingFace and place in weights/

### 4. Run the API
uvicorn api.main:app --reload

### 5. Open Swagger docs
http://localhost:8000/docs

## Training
Configure hyperparameters in configs/config.yaml then run:
python src/train.py

## Experiments
W&B dashboard: https://wandb.ai/hakobyananna2002-no-company/plant-disease

## Model Weights
Download from HuggingFace: https://huggingface.co/annehakobyan/plant-disease-classifier


## Live API
Swagger UI: https://annehakobyan-plant-disease-api.hf.space/docs


## Experiments
W&B Report: https://api.wandb.ai/links/hakobyananna2002-no-company/hl89he24