import torch
import torch.nn as nn
from vision_encoder import CXRVisionEncoder
from text_decoder import RadiologyReportDecoder

class MedicalReportGenerator(nn.Module):
    """
    Combines the Vision Encoder (DenseNet) and Language Decoder (BioClinicalBERT).
    Takes a CXR image and outputs autoregressive text logits.
    """
    def __init__(self, cnn_dim=1024, text_dim=768, max_length=128):
        super(MedicalReportGenerator, self).__init__()
        
        # 1. Vision Backbone
        self.vision_encoder = CXRVisionEncoder()
        
        # 2. Projection Layer 
        # DenseNet gives 1024-dim features. ClinicalBERT expects 768-dim for cross-attention.
        self.visual_projection = nn.Linear(cnn_dim, text_dim)
        
        # 3. Text Backbone
        self.text_decoder = RadiologyReportDecoder(max_length=max_length)
        
    def forward(self, images, input_ids, attention_mask):
        """
        Inputs:
            images: (B, NumViews, 3, H, W) - our dataset gives (B, 3, 3, 224, 224)
            input_ids: (B, SeqLen)
            attention_mask: (B, SeqLen)
        """
        B, NumViews, C, H, W = images.size()
        
        # 1. Flatten Batch & Views to process all images in one CNN pass
        # Shape: (B * NumViews, 3, H, W)
        flat_images = images.view(-1, C, H, W)
        
        # 2. Extract spatial patches
        # Shape: (B * NumViews, NumPatches, 1024)
        flat_features = self.vision_encoder(flat_images)
        
        # 3. Reshape back and Concatenate Views
        # Shape: (B, NumViews, NumPatches, 1024)
        view_features = flat_features.view(B, NumViews, -1, flat_features.size(-1))
        
        # Concatenate: (B, NumViews * NumPatches, 1024)
        # This gives the transformer the full spatial context of all 3 views
        aggregated_features = view_features.reshape(B, -1, flat_features.size(-1))
        
        # 4. Project aggregated visual features to text dimension
        projected_features = self.visual_projection(aggregated_features)
        
        # 5. Decode into text 
        sequence_logits = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=projected_features
        )
        
        return sequence_logits

    def generate(self, images, tokenizer, max_length=128, beam_size=5, device='cpu'):
        """
        Advanced Generation using Beam Search with Multi-View Aggregation.
        images shape: (B, NumViews, 3, H, W)
        """
        self.eval()
        with torch.no_grad():
            B, NumViews, C, H, W = images.size()
            
            # 1. Multi-View Encoding
            flat_images = images.view(-1, C, H, W).to(device)
            flat_features = self.vision_encoder(flat_images)
            
            # Concatenate spatial patches from all views
            view_features = flat_features.view(B, NumViews, -1, flat_features.size(-1))
            aggregated_features = view_features.reshape(B, -1, flat_features.size(-1))
            
            projected_features = self.visual_projection(aggregated_features)
            
            # 2. Setup Beam Search (Batch size 1 support only for inference)
            start_token = tokenizer.cls_token_id
            sep_token = tokenizer.sep_token_id
            
            beams = [(0.0, torch.tensor([[start_token]], device=device))]
            
            for _ in range(max_length):
                new_beams = []
                for score, seq in beams:
                    if seq[0, -1].item() == sep_token:
                        new_beams.append((score, seq))
                        continue
                    
                    logits = self.text_decoder(
                        input_ids=seq,
                        attention_mask=torch.ones_like(seq),
                        encoder_hidden_states=projected_features
                    )
                    
                    log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                    top_probs, top_ids = torch.topk(log_probs, beam_size)
                    
                    for i in range(beam_size):
                        new_score = score + top_probs[i].item()
                        new_seq = torch.cat([seq, top_ids[i].unsqueeze(0).unsqueeze(0)], dim=-1)
                        new_beams.append((new_score, new_seq))
                
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_size]
                
                if all(b[1][0, -1].item() == sep_token for b in beams):
                    break
            
            best_seq = beams[0][1]
            return tokenizer.decode(best_seq.squeeze(0), skip_special_tokens=True)


if __name__ == "__main__":
    # Integration Test
    generator = MedicalReportGenerator()
    
    # Dummy Batch
    b_images = torch.randn(2, 3, 224, 224)
    b_input_ids = torch.tensor([
        [101, 2023, 2003, 1037, 3231, 102],
        [101, 2045, 2003, 1042,  102,   0]  # Padded
    ])
    b_att_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0]
    ])
    
    print("Running forward pass...")
    logits = generator(b_images, b_input_ids, b_att_mask)
    print(f"Final Output Logits Shape: {logits.shape}")
