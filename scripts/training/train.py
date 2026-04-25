import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

# Path configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_prep')))

from multimodal_generator import MedicalReportGenerator
from dataset import MedicalReportDataset

def train():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    epochs = 20
    accumulation_steps = 4
    max_length = 512
    
    train_csv = "data/processed/mimic_cxr_aug_train_cleaned.csv"
    val_csv = "data/processed/mimic_cxr_aug_validate_cleaned.csv"
    images_root = "data/images/official_data_iccv_final"
    checkpoint_path = "models/checkpoints/best_model.pth"
    
    os.makedirs("models/checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir="models/logs")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = MedicalReportGenerator(max_length=max_length).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*30}")
    print(f"Model Summary:")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*30}\n")

    # 2. Differential Learning Rates (LOWERED for stability)
    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if 'crossattention' in name or 'visual_projection' in name:
            new_params.append(param)
        else:
            pretrained_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': 1e-5}, # Lowered from 2e-5
        {'params': new_params, 'lr': 4e-4}         # Lowered from 1e-4
    ], weight_decay=0.01)
    
    # 3. Resume Logic
    best_val_loss = float('inf')
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Restored optimizer state.")
            except:
                print("Could not restore optimizer state. Starting fresh optimizer.")
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Resuming from Epoch {start_epoch} | Best Val Loss: {best_val_loss:.4f}")
        else:
            # Handle old state-dict-only checkpoints
            model.load_state_dict(checkpoint)
            print("Loaded weights (Legacy Format). Optimizer reset.")

    # 4. Data
    train_dataset = MedicalReportDataset(train_csv, images_root, tokenizer, max_length=max_length)
    val_dataset = MedicalReportDataset(val_csv, images_root, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Training
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    scaler = torch.amp.GradScaler('cuda')
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(pbar):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.amp.autocast('cuda'):
                logits = model(images, input_ids[:, :-1], attention_mask[:, :-1])
                targets = input_ids[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            if i % 10 == 0:
                writer.add_scalar("Loss/train", loss.item() * accumulation_steps, global_step)
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})
            global_step += 1
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(images, input_ids[:, :-1], attention_mask[:, :-1])
                    targets = input_ids[:, 1:].contiguous()
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New Best Model! Saving checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)

    writer.close()

if __name__ == "__main__":
    train()
