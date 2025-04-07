from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import AutoModelForImageClassification, AutoImageProcessor
import io
import torch.nn.functional as F

app = FastAPI()

# Allow CORS from all origins (you can customize this if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, use specific URLs for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Load the pre-trained ViT model
model_path = "vit_liveness_detection_modelV2.pth"  # Change to your actual model file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with ignore_mismatched_sizes=True
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=5,  # Match your trained model's number of classes
    ignore_mismatched_sizes=True  # Ignore classifier layer size mismatch
)

# Load the trained weights
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

# Move model to GPU/CPU and set to evaluation mode
model.to(device)
model.eval()

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Load model and processor for version 2.2.3
MODEL_NAME_V223 = "nguyenkhoa/dinov2_Liveness_detection_v2.2.3"
liveness_model_v223 = AutoModelForImageClassification.from_pretrained(MODEL_NAME_V223)
liveness_processor_v223 = AutoImageProcessor.from_pretrained(MODEL_NAME_V223)

# Load model and processor for version 2.1.4
MODEL_NAME_V214 = "nguyenkhoa/dinov2_Liveness_detection_v2.1.4"
liveness_model_v214 = AutoModelForImageClassification.from_pretrained(MODEL_NAME_V214)
liveness_processor_v214 = AutoImageProcessor.from_pretrained(MODEL_NAME_V214)

def predict_image(model, processor, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    class_labels = model.config.id2label
    predicted_class = class_labels[predicted_class_idx]
    return predicted_class

# Preprocessing function
def preprocess_image(image: Image.Image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(device)


@app.post("/model/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess_image(image)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1).squeeze().tolist()  # Convert to list

    # Get predicted class
    predicted_class = torch.argmax(logits, dim=-1).item()

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities  # List of probabilities for each class
    }

@app.post("/model/predict/dinov2/v2.2.3")
async def predict_v223(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class = predict_image(liveness_model_v223, liveness_processor_v223, image_bytes)
    return {"version": "v2.2.3", "filename": file.filename, "predicted_class": predicted_class}

@app.post("/model/predict/dinov2/v2.1.4")
async def predict_v214(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class = predict_image(liveness_model_v214, liveness_processor_v214, image_bytes)
    return {"version": "v2.1.4", "filename": file.filename, "predicted_class": predicted_class}

@app.get("/model/predict")
async def predict():
    # Your model prediction logic
    return {"predicted_class": "test"}
