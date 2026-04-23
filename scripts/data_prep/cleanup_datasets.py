import pandas as pd
import ast
import os

def process_and_clean_dataset(input_csv, output_csv, base_img_dir='../../data/images/official_data_iccv_final'):
    print(f"\nProcessing {input_csv}...")
    df = pd.read_csv(input_csv)
    original_len = len(df)
    
    # 1. Drop redundant index columns
    cols_to_drop = [c for c in df.columns if 'Unnamed' in c]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"  Dropped columns: {cols_to_drop}")

    # 2. Parse stringified lists
    array_cols = ['image', 'view', 'AP', 'PA', 'Lateral', 'text', 'text_augment']
    for col in array_cols:
        if col in df.columns:
            # Safely evaluate string representation of lists
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    
    # 3. Handle 'nan' strings within the view arrays
    if 'view' in df.columns:
        df['view'] = df['view'].apply(lambda views: [v for v in views if str(v).lower() != 'nan'] if isinstance(views, list) else views)

    # 4. Filter missing images
    print("  Checking image existence...")
    valid_rows = []
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0 and idx > 0:
            print(f"    Checked {idx}/{original_len} rows...")
            
        all_images_exist = True
        if isinstance(row.get('image'), list):
            for img_path in row['image']:
                full_path = os.path.join(base_img_dir, img_path)
                if not os.path.exists(full_path):
                    all_images_exist = False
                    break
        else:
            all_images_exist = False
            
        valid_rows.append(all_images_exist)
            
    df = df[valid_rows]
    final_len = len(df)
    
    print(f"  Original rows: {original_len}")
    print(f"  Final valid rows: {final_len}")
    print(f"  Dropped {original_len - final_len} rows due to missing images.")
    
    # Save cleaned
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned dataset to: {output_csv}")

if __name__ == "__main__":
    process_and_clean_dataset('../../data/raw/mimic_cxr_aug_validate.csv', '../../data/processed/mimic_cxr_aug_validate_cleaned.csv')
    process_and_clean_dataset('../../data/raw/mimic_cxr_aug_train.csv', '../../data/processed/mimic_cxr_aug_train_cleaned.csv')
