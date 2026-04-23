import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class RadiologyReportDecoder(nn.Module):
    """
    Language Decoder using ClinicalBERT or similar HuggingFace model 
    adapted for autoregressive generation with cross-attention.
    """
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", max_length=128):
        super(RadiologyReportDecoder, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # We need a model capable of Seq2Seq (Cross Attention)
        # Using a standard BERT wrapper from huggingface as a Decoder
        # by setting is_decoder=True and add_cross_attention=True
        
        from transformers import BertConfig, BertLMHeadModel
        config = BertConfig.from_pretrained(model_name)
        config.is_decoder = True
        config.add_cross_attention = True
        config.tie_word_embeddings = False
        
        self.decoder = BertLMHeadModel.from_pretrained(model_name, config=config)
        self.max_length = max_length
        
    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        """
        Inputs:
            input_ids: (B, seq_len) token ids of the radiology report
            attention_mask: (B, seq_len) mask for padding
            encoder_hidden_states: (B, num_patches, hidden_dim) the spatial grid from Vision Encoder
        """
        # BertLMHeadModel automatically handles the cross attention 
        # when encoder_hidden_states are provided.
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states
        )
        return outputs.logits

if __name__ == "__main__":
    # Test
    decoder = RadiologyReportDecoder()
    dummy_input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]]) # [CLS] this is a test [SEP]
    dummy_att_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
    
    # Simulate vision encoder output (Batch=1, Patches=49, Dim=1024)
    # Note: ClinicalBERT's hidden size is 768. 
    # We will need a projection layer in the Fusion module to map 1024 -> 768!
    dummy_encoder_states = torch.randn(1, 49, 768) 
    
    logits = decoder(dummy_input_ids, dummy_att_mask, dummy_encoder_states)
    print(f"Logits shape: {logits.shape} (Expected: Batch, SeqLen, VocabSize)")
