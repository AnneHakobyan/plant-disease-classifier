from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import timm
import json
from PIL import Image
import torchvision.transforms as T
import io

app = FastAPI(
    title="Plant Disease Classifier",
    description="""
    Identifies plant diseases from leaf images.
    Returns the disease label only (e.g. 'late blight'),
    regardless of which host plant the disease appears on.
    """,
    version="1.0.0"
)

# ── Global model state ─────────────────────────────────────────
model       = None
transform   = None
idx_to_class = None


@app.on_event("startup")
async def load_model():
    global model, transform, idx_to_class

    # Download weights from HuggingFace if not present
    from api.download_model import download_weights
    download_weights()

    # Load class map
    with open("configs/class_map.json") as f:
        class_map = json.load(f)
    idx_to_class = class_map["idx_to_class"]
    num_classes  = len(idx_to_class)

    # Load model
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load("weights/best_model.pt", map_location="cpu"))
    model.eval()

    transform = T.Compose([
        T.Resize(330),
        T.CenterCrop(300),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"Model loaded. {num_classes} classes.")

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "classes": len(idx_to_class)}


@app.post("/predict", summary="Classify plant disease from image")
async def predict(file: UploadFile = File(..., description="Plant leaf image (JPG/PNG)")):
    """
    Upload a plant leaf image and get the disease prediction.

    - **Input**: image file (JPG or PNG)
    - **Output**: disease name, confidence score, and top 3 predictions
    """
    # Validate file type
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported.")

    # Read and preprocess
    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    tensor = transform(img).unsqueeze(0)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top3_indices = probs.topk(3).indices.tolist()

    return JSONResponse({
        "disease"   : idx_to_class[str(probs.argmax().item())],
        "confidence": round(probs.max().item(), 4),
        "top_3"     : [
            {
                "disease"   : idx_to_class[str(i)],
                "confidence": round(probs[i].item(), 4)
            }
            for i in top3_indices
        ]
    })