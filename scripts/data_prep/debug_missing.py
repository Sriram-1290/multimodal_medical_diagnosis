import pandas as pd
import ast
import os

# Paths relative to repository root
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/mimic_cxr_aug_validate.csv'))
images_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/images/official_data_iccv_final'))

if not os.path.exists(csv_path):
    # Fallback to current directory check
    csv_path = 'mimic_cxr_aug_validate.csv'
    images_root = 'official_data_iccv_final'

print(f"Using CSV path: {csv_path}")
print(f"Using Images root: {images_root}")

try:
    df = pd.read_csv(csv_path)
    missing_count = 0
    for idx, row in df.iterrows():
        if missing_count >= 5:
            break
        images = ast.literal_eval(row['image'])
        for img in images:
            path = os.path.join(images_root, img)
            if not os.path.exists(path):
                print(f"MISSING: {path}")
                print(f"List dir of parent {os.path.dirname(path)}:")
                if os.path.exists(os.path.dirname(path)):
                    print(os.listdir(os.path.dirname(path)))
                else:
                    print("Parent dir also missing!")
                missing_count += 1
                break
    if missing_count == 0:
        print("Checked dataset. No missing images found in first rows.")
except Exception as e:
    print(f"Error checking missing images: {e}")
