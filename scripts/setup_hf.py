import os
import sys
from transformers import AutoTokenizer, BertLMHeadModel, BertConfig

def setup():
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    print(f"Pre-caching model and tokenizer for: {model_name}")
    
    # 1. Tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Model & Config
    print("Downloading model weights and config...")
    # Loading once with from_pretrained ensures local cache is populated
    # We use BertLMHeadModel because that's what the decoder uses
    config = BertConfig.from_pretrained(model_name)
    config.is_decoder = True
    config.add_cross_attention = True
    config.tie_word_embeddings = False
    
    model = BertLMHeadModel.from_pretrained(model_name, config=config)
    
    print("\n[SUCCESS] Model and tokenizer are successfully cached.")
    print("You can now run the training script without background downloads.")

if __name__ == "__main__":
    # Silence warnings during setup
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    setup()
