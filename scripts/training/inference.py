import torch
import torch.nn as nn
import os
import sys

# Add the scripts/models path to search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from multimodal_generator import MedicalReportGenerator
from transformers import AutoTokenizer

if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = MedicalReportGenerator().to(device)
    
    # Load dummy image (Batch=1, NumViews=3, C=3, H=224, W=224)
    dummy_img = torch.randn(1, 3, 3, 224, 224).to(device)
    
    print("Generating report (Beam Search)...")
    report = model.generate(dummy_img, tokenizer, device=device, beam_size=5)
    print(f"Generated Report: {report}")
