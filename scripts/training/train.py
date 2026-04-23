import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the scripts/models path to search path so we can import them
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_prep')))

from multimodal_generator import MedicalReportGenerator
from dataset import MedicalReportDataset
from transformers import AutoTokenizer

from torch.utils.tensorboard import SummaryWriter

def train():
    # 1. Hyperparameters & Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 2  # Reduced batch size for multi-view memory usage
    epochs = 20
    lr = 5e-5
    max_length = 512
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    
    # Paths
    train_csv = "data/processed/mimic_cxr_aug_train_cleaned.csv"
    val_csv = "data/processed/mimic_cxr_aug_validate_cleaned.csv"
    images_root = "data/images/official_data_iccv_final"
    checkpoint_dir = "models/checkpoints"
    log_dir = "models/logs"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 2. Components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MedicalReportGenerator(max_length=max_length).to(device)
    
    # Dataset & DataLoader
    train_dataset = MedicalReportDataset(train_csv, images_root, tokenizer, max_length=max_length)
    val_dataset = MedicalReportDataset(val_csv, images_root, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in pbar:
            images = batch['image'].to(device) # (B, 3, 3, 224, 224)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Autoregressive shift
            logits = model(images, input_ids[:, :-1], attention_mask[:, :-1])
            targets = input_ids[:, 1:].contiguous()
            
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            pbar.set_postfix({'loss': loss.item()})
            global_step += 1
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 4. Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                logits = model(images, input_ids[:, :-1], attention_mask[:, :-1])
                targets = input_ids[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 5. Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print("Checkpoint saved!")
            
    # Final save
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last_model.pth"))
    writer.close()

if __name__ == "__main__":
    train()
