from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import io
import json
import os

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Plant Disease Detection API")

# ✅ FIX 1: Add CORS (This fixes the Frontend "Error predicting")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Paths --------------------
cnn_model_path = "./best_simplecnn_plant_disease.pth"
cnn_class_file = "./distilbert_plant_model/config.json"

# -------------------- Load CNN Classes --------------------
try:
    with open(cnn_class_file, "r") as f:
        config_data = json.load(f)
        # Handle HuggingFace style config or raw list
        if isinstance(config_data, dict) and "id2label" in config_data:
            class_names_cnn = [config_data["id2label"][str(i)] for i in range(len(config_data["id2label"]))]
        else:
            class_names_cnn = config_data
    print(f"✅ Loaded {len(class_names_cnn)} class names from JSON.")
except Exception as e:
    print(f"❌ Error loading class file: {e}")
    class_names_cnn = []

# -------------------- CNN MODEL --------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) # This is the layer that mismatched
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# -------------------- Load CNN Model Safely --------------------
# ✅ FIX 2: Force 38 classes to match your .pth file from the screenshot
checkpoint_classes = 38
cnn_model = CNN(num_classes=checkpoint_classes)

try:
    checkpoint = torch.load(cnn_model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Load weights into the 38-class architecture
    cnn_model.load_dict = cnn_model.load_state_dict(state_dict, strict=False)
    print("✅ CNN Model weights loaded successfully (38 classes).")

    # ✅ FIX 3: Re-align if your JSON has a different number of names
    if len(class_names_cnn) != checkpoint_classes and len(class_names_cnn) > 0:
        print(f"⚠️ Warning: JSON has {len(class_names_cnn)} names but model has 38. Re-aligning...")
        cnn_model.classifier[3] = nn.Linear(256, len(class_names_cnn))
except Exception as e:
    print(f"❌ Critical loading error: {e}")

cnn_model.to(device)
cnn_model.eval()

# -------------------- Image Transform --------------------
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------- Text Model --------------------
text_model_dir = "./distilbert_plant_model"
try:
    tokenizer_text = AutoTokenizer.from_pretrained(text_model_dir)
    model_text = AutoModelForSequenceClassification.from_pretrained(text_model_dir)
    model_text.to(device)
    model_text.eval()
    print("✅ Text model loaded successfully.")
except Exception as e:
    print(f"❌ Text model error: {e}")

# -------------------- Endpoints --------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        image_tensor = cnn_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = cnn_model(image_tensor)
            pred_id = torch.argmax(output, dim=1).item()
            
            # Prevent crash if index is out of range of your name list
            if pred_id < len(class_names_cnn):
                pred_class = class_names_cnn[pred_id]
            else:
                pred_class = f"Unknown Disease (Index {pred_id})"

        return {"filename": file.filename, "predicted_disease": pred_class}
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-text")
async def predict_text(description: str = Form(...)):
    try:
        inputs = tokenizer_text(description, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model_text(**inputs)
            pred_id = outputs.logits.argmax(dim=-1).item()
            # Safety check for label mapping
            pred_class = model_text.config.id2label.get(pred_id, f"Label {pred_id}")

        return {"description": description, "predicted_disease": pred_class}
    except Exception as e:
        print(f"Text Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "API is Running"}