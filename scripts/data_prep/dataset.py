import os
import ast
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MedicalReportDataset(Dataset):
    """
    Dataset for loading CXR images and radiology reports.
    """
    def __init__(self, csv_path, images_root, tokenizer, max_length=128, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Handle Images (Load all unique views)
        img_paths = row['image']
        if isinstance(img_paths, str):
            img_paths = ast.literal_eval(img_paths)
            
        # Standardize: Take up to 3 images per study (AP, PA, Lateral)
        # We fill with zeros if there are fewer than 3 images for batching consistency
        max_views = 3
        study_images = []
        
        for i in range(max_views):
            if i < len(img_paths):
                img_path = img_paths[i]
                full_path = os.path.join(self.images_root, img_path)
                try:
                    img = Image.open(full_path).convert('RGB')
                    img = self.transform(img)
                    study_images.append(img)
                except Exception:
                    # Fallback if image load fails
                    study_images.append(torch.zeros(3, 224, 224))
            else:
                # Pad study with zeroes
                study_images.append(torch.zeros(3, 224, 224))
        
        # Stack into (3, 3, 224, 224)
        images_tensor = torch.stack(study_images)
        
        # 2. Handle Text
        text_list = row['text']
        if isinstance(text_list, str):
            text_list = ast.literal_eval(text_list)
        
        report = text_list[0] if len(text_list) > 0 else ""
        
        tokens = self.tokenizer(
            report,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': images_tensor, # (3, 3, 224, 224)
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    csv_file = "data/processed/mimic_cxr_aug_validate_cleaned.csv"
    img_dir = "data/images/official_data_iccv_final"
    
    if os.path.exists(csv_file):
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        dataset = MedicalReportDataset(csv_file, img_dir, tokenizer)
        print(f"Dataset length: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        
        # Decode back to check
        print(f"Sample Report: {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)}")
    else:
        print(f"File {csv_file} not found. Run cleanup first.")
