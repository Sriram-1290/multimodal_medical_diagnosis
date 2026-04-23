import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from typing import List, Optional
import io

# Add models path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))

from multimodal_generator import MedicalReportGenerator
from transformers import AutoTokenizer

app = FastAPI(title="Medical Diagnosis AI API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "models/checkpoints/best_model.pth"
TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"

def load_resources():
    global model, tokenizer
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Model will not be loaded.")
        return False
    
    try:
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        model = MedicalReportGenerator().to(device)
        
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Initial load attempt
model_loaded = load_resources()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/api/status")
async def get_status():
    """Check if the medical model is loaded and ready."""
    return {
        "status": "ready" if model_loaded else "model_not_found",
        "device": str(device),
        "checkpoint_exists": os.path.exists(CHECKPOINT_PATH)
    }

@app.post("/api/predict")
async def predict(
    ap_view: Optional[UploadFile] = File(None),
    pa_view: Optional[UploadFile] = File(None),
    lateral_view: Optional[UploadFile] = File(None)
):
    """
    Generate a radiology report from up to three CXR views.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model checkpoint not found. Service unavailable.")

    # We need at least one view
    uploaded_files = [ap_view, pa_view, lateral_view]
    if all(f is None for f in uploaded_files):
        raise HTTPException(status_code=400, detail="At least one image view must be provided.")

    try:
        # Prepare 3 views (fill missing with zero tensors)
        images_list = []
        for file in uploaded_files:
            if file:
                content = await file.read()
                img = Image.open(io.BytesIO(content)).convert('RGB')
                images_list.append(transform(img))
            else:
                images_list.append(torch.zeros(3, 224, 224))
        
        # Shape: (1, 3, 3, 224, 224) -> Batch=1, NumViews=3, C=3, H, W
        input_tensor = torch.stack(images_list).unsqueeze(0).to(device)

        # Generate report
        print("Running inference...")
        report = model.generate(input_tensor, tokenizer, k=5, repetition_penalty=2.0)
        
        return {
            "report": report,
            "views_received": [f.filename for f in uploaded_files if f]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Mount static files for the frontend
static_path = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)

app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
