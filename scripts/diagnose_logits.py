import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys

# Add models path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'scripts/models')))
from multimodal_generator import MedicalReportGenerator
from transformers import AutoTokenizer

def check_logits_variation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "models/checkpoints/best_model.pth"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = MedicalReportGenerator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Two different image inputs
    img1 = torch.randn(1, 3, 3, 224, 224).to(device)
    img2 = torch.ones(1, 3, 3, 224, 224).to(device) * -1.0 # Very different

    with torch.no_grad():
        # Start token [CLS]
        input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)
        
        def get_logits(img):
            # Same logic as forward/generate
            flat_images = img.view(-1, 3, 224, 224)
            flat_features = model.vision_encoder(flat_images)
            view_features = flat_features.reshape(1, 3, -1, 1024)
            agg_features = view_features.reshape(1, -1, 1024)
            proj_features = model.visual_projection(agg_features)
            
            logits = model.text_decoder(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                encoder_hidden_states=proj_features
            )
            return logits

        l1 = get_logits(img1)
        l2 = get_logits(img2)
        
        diff = torch.abs(l1 - l2).mean().item()
        print(f"Logits diff (Mean Abs): {diff:.8f}")
        
        if diff < 1e-6:
            print("CRITICAL: Logits are identical! Visual features are ignored.")
        else:
            print(f"Logits differ. (Sample 1 Top token: {torch.argmax(l1[0, -1, :]).item()})")
            print(f"(Sample 2 Top token: {torch.argmax(l2[0, -1, :]).item()})")

if __name__ == "__main__":
    check_logits_variation()
