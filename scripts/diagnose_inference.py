import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys
from PIL import Image
from torchvision import transforms
import io

# Add models path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'scripts/models')))
from multimodal_generator import MedicalReportGenerator
from transformers import AutoTokenizer

def test_inference_difference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "models/checkpoints/best_model.pth"
    tokenizer_name = "emilyalsentzer/Bio_ClinicalBERT"

    print("Loading resources...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = MedicalReportGenerator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create two different dummy images (or use real ones if available, but dummy is fine to check if any diff exists)
    img1 = torch.randn(3, 224, 224)
    img2 = torch.ones(3, 224, 224) * 0.5 # significantly different

    def get_report(img):
        # (1, 3, 3, 224, 224)
        input_tensor = torch.stack([img, torch.zeros_like(img), torch.zeros_like(img)]).unsqueeze(0).to(device)
        report = model.generate(input_tensor, tokenizer, device=device, beam_size=1) # Greedy for speed
        return report

    print("Inference 1...")
    report1 = get_report(img1)
    print(f"Report 1: {report1}")

    print("Inference 2...")
    report2 = get_report(img2)
    print(f"Report 2: {report2}")

    if report1 == report2:
        print("\nFAILURE: Reports are identical despite different inputs.")
    else:
        print("\nSUCCESS: Reports are different.")

if __name__ == "__main__":
    test_inference_difference()
