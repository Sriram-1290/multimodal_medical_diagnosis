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
        
    def encode_images(self, images):
        """Helper to extract and project visual features from multiple views."""
        B, NumViews, C, H, W = images.size()
        flat_images = images.view(-1, C, H, W)
        flat_features = self.vision_encoder(flat_images)
        view_features = flat_features.view(B, NumViews, -1, flat_features.size(-1))
        aggregated_features = view_features.reshape(B, -1, flat_features.size(-1))
        return self.visual_projection(aggregated_features)

    def forward(self, images, input_ids, attention_mask):
        projected_features = self.encode_images(images)
        sequence_logits = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=projected_features
        )
        return sequence_logits

    def generate(self, images, tokenizer, k=5, max_length=128, repetition_penalty=1.5):
        """
        Advanced Generation using Beam Search with Repetition Penalty.
        k: beam width
        repetition_penalty: >1.0 to discourage repeating tokens
        """
        self.eval()
        device = images.device
        
        with torch.no_grad():
            encoder_hidden_states = self.encode_images(images)
            start_token = tokenizer.cls_token_id
            
            # (score, sequence, finished_flag)
            beams = [(0.0, torch.tensor([[start_token]], device=device), False)]
            
            for _ in range(max_length):
                new_beams = []
                all_finished = True
                
                for score, seq, finished in beams:
                    if finished:
                        new_beams.append((score, seq, True))
                        continue
                    
                    all_finished = False
                    logits = self.text_decoder(
                        input_ids=seq,
                        attention_mask=torch.ones_like(seq),
                        encoder_hidden_states=encoder_hidden_states
                    )
                    
                    next_token_logits = logits[0, -1, :].clone()
                    
                    # Apply repetition penalty to already generated tokens
                    # This prevents the ": : : :" looping
                    generated_tokens = set(seq[0].tolist())
                    for token_id in generated_tokens:
                        if next_token_logits[token_id] > 0:
                            next_token_logits[token_id] /= repetition_penalty
                        else:
                            next_token_logits[token_id] *= repetition_penalty
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    top_probs, top_ids = torch.topk(probs, k)
                    
                    for i in range(k):
                        token_id = top_ids[i].view(1, 1)
                        new_score = score + torch.log(top_probs[i]).item()
                        new_seq = torch.cat([seq, token_id], dim=1)
                        is_done = (token_id.item() == tokenizer.sep_token_id)
                        new_beams.append((new_score, new_seq, is_done))
                
                if all_finished:
                    break
                    
                # Keep top k beams
                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]
            
            # Pick best complete beam
            best_seq = beams[0][1]
            return tokenizer.decode(best_seq.squeeze(0), skip_special_tokens=True)

if __name__ == "__main__":
    generator = MedicalReportGenerator()
    b_images = torch.randn(1, 3, 3, 224, 224)
    print("Running forward pass...")
    # ... test logic ...
