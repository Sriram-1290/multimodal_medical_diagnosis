import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys

# Add models path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'scripts/models')))
from multimodal_generator import MedicalReportGenerator

def check_feature_variation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "models/checkpoints/best_model.pth"
    
    model = MedicalReportGenerator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    img1 = torch.randn(1, 3, 3, 224, 224).to(device)
    img2 = torch.ones(1, 3, 3, 224, 224).to(device)

    with torch.no_grad():
        # Mimic part of generate()
        f1 = model.vision_encoder(img1.view(-1, 3, 224, 224))
        f2 = model.vision_encoder(img2.view(-1, 3, 224, 224))
        
        v1 = model.visual_projection(f1.view(1, -1, 1024))
        v2 = model.visual_projection(f2.view(1, -1, 1024))
        
        print(f"Features 1 (mean): {v1.mean().item():.6f}")
        print(f"Features 2 (mean): {v2.mean().item():.6f}")
        print(f"Difference (MSE): {torch.nn.functional.mse_loss(v1, v2).item():.6f}")

if __name__ == "__main__":
    check_feature_variation()
