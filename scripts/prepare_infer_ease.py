import os
import pandas as pd
import shutil
import ast

def prepare_precise_samples():
    csv_path = "data/processed/mimic_cxr_aug_validate_cleaned.csv"
    src_images_root = "data/images/official_data_iccv_final"
    dest_root = "data/infer_ease"
    
    # 1. Wipe and recreate
    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)
    os.makedirs(dest_root, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    # Take 10 samples
    samples = df.head(10)
    
    for i, (_, row) in enumerate(samples.iterrows()):
        sample_num = i + 1
        sample_dir = os.path.join(dest_root, f"sample_{sample_num}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 2. Extract mappings
        image_paths = ast.literal_eval(row['image'])
        view_names = ast.literal_eval(row['view'])
        
        # We want to map them precisely
        view_map = {}
        for img_path, v_name in zip(image_paths, view_names):
            v_lower = v_name.lower()
            if 'pa' in v_lower and 'pa' not in view_map:
                view_map['pa'] = img_path
            elif 'ap' in v_lower and 'ap' not in view_map:
                view_map['ap'] = img_path
            elif ('lateral' in v_lower or v_lower == 'll') and 'lateral' not in view_map:
                view_map['lateral'] = img_path

        # 3. Copy with new names
        for v_type, img_rel_path in view_map.items():
            src_path = os.path.join(src_images_root, img_rel_path)
            dest_path = os.path.join(sample_dir, f"{v_type}.jpg")
            
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                print(f"Sample {sample_num}: Copied {v_type}")
            else:
                print(f"Warning: {src_path} missing")

        # 4. Save Ground Truth Report
        report_list = ast.literal_eval(row['text'])
        report = report_list[0] if report_list else "No report available."
        with open(os.path.join(sample_dir, "ground_truth.txt"), "w") as f:
            f.write(report)
            
    print(f"\nDone! Precise samples created in {dest_root}")

if __name__ == "__main__":
    prepare_precise_samples()
