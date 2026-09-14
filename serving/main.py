"""
FastAPI serving script for the Flowers102 EfficientNet-B1 classifier.

Loads a trained checkpoint (checkpoints/best.pt, produced by the training
notebook) and exposes a single POST /predict/ endpoint that accepts an
uploaded image and returns the predicted flower class.

Run:
    uvicorn serving.main:app --reload

Test:
    curl -X POST "http://localhost:8000/predict/" -F "file=@path_to_image.jpg"
"""

import io
import json
import urllib.request

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b1

app = FastAPI(title="Flowers102 EfficientNet-B1 Classifier")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 102
CHECKPOINT_PATH = "checkpoints/best.pt"

# Class name mapping is fetched from the same verified public source used in
# the training notebook, rather than a hand-typed list (see notebook §3 for
# why: a manually transcribed list was found to contain 104 entries instead
# of 102 during verification).
CAT_TO_NAME_URL = (
    "https://raw.githubusercontent.com/udacity/aipnd-project/master/cat_to_name.json"
)
with urllib.request.urlopen(CAT_TO_NAME_URL) as resp:
    _cat_to_name = json.load(resp)
assert len(_cat_to_name) == NUM_CLASSES
CLASS_NAMES = [_cat_to_name[str(i)] for i in range(1, NUM_CLASSES + 1)]

# Model architecture must match the training notebook exactly, otherwise
# load_state_dict will fail (or silently load into the wrong shape).
model = efficientnet_b1(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(num_ftrs, NUM_CLASSES),
)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Must match the "eval" transform used during training/validation, not the
# augmented "train" transform.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    tensor = transform(image)
    return tensor.unsqueeze(0).to(DEVICE)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)

        return JSONResponse(
            content={
                "predicted_class": CLASS_NAMES[top_idx.item()],
                "confidence": round(top_prob.item(), 4),
            }
        )
    except Exception as e:  # noqa: BLE001 - return a clean 400 instead of crashing
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
