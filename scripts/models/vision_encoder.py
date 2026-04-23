import torch
import torch.nn as nn
import torchvision.models as models

class CXRVisionEncoder(nn.Module):
    """
    Vision Encoder using Pretrained DenseNet-121.
    Extracts spatial feature maps rather than a global pooled vector, 
    allowing a downstream Language Decoder to use Cross-Attention over different regions of the image.
    """
    def __init__(self, pretrained=True, freeze_weights=False):
        super(CXRVisionEncoder, self).__init__()
        
        # Load standard ImageNet-pretrained DenseNet-121
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)
        
        # Remove the final classification head and pooling layer
        # Output shape of densenet.features will be (Batch, 1024, H/32, W/32)
        # For a 224x224 image, output is (Batch, 1024, 7, 7)
        self.feature_extractor = densenet.features
        self.out_channels = 1024
        
        if freeze_weights:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
    def forward(self, images):
        """
        Inputs:
            images: Tensor of shape (B, C, H, W) where C=3
            
        Outputs:
            features: Tensor of shape (B, 1024, H', W') spatial grid.
                      We will reshape to (B, H'*W', 1024) so it can act as a sequence
                      for the Transformer Cross-Attention.
        """
        # B x 1024 x 7 x 7 (if H, W = 224, 224)
        spatial_features = self.feature_extractor(images)
        
        B, C, H_out, W_out = spatial_features.size()
        
        # Flatten spatial dimensions: B x C x (H_out * W_out) => B x (H_out * W_out) x C
        # Example: B x 1024 x 49 => B x 49 x 1024
        # This treats the 49 spatial patches as "tokens" for the language model to attend to
        seq_features = spatial_features.view(B, C, -1).permute(0, 2, 1)
        
        return seq_features

if __name__ == "__main__":
    # Quick Test
    encoder = CXRVisionEncoder()
    dummy_input = torch.randn(2, 3, 224, 224) # Batch size 2, 3 channels, 224x224
    output = encoder(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Expected: B, 49, 1024)")
