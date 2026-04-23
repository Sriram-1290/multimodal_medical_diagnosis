import pandas as pd
import ast
import os

def check_data_quality(filepath):
    print(f"\n{'='*40}")
    print(f"--- Quality Check for {filepath} ---")
    try:
        df = pd.read_csv(filepath)
        
        # 1. Check for redundant index columns
        unnamed_cols = [c for c in df.columns if 'Unnamed' in c]
        if unnamed_cols:
            print(f"Redundant index columns found: {unnamed_cols}")
            
        # 2. Check for missing values
        print("\nMissing Values per column:")
        print(df.isnull().sum())
        
        # 3. Check data types of array-like columns
        array_cols = ['image', 'view', 'AP', 'PA', 'Lateral', 'text', 'text_augment']
        print("\nChecking if array-like columns are stringified lists:")
        for col in array_cols:
            if col in df.columns:
                sample_val = df.iloc[0][col]
                print(f"  {col} type: {type(sample_val).__name__} (starts with '[': {str(sample_val).startswith('[')})")
        
        # 4. Check 'nan' string values in view or text
        nan_string_views = df[df['view'].astype(str).str.contains("'nan'", na=False)]
        print(f"\nRows with \"'nan'\" string in 'view': {len(nan_string_views)}")
        
        # 5. Check missing files (sample 100 random rows to be fast)
        missing_files = 0
        total_files_checked = 0
        
        sample_df = df.sample(min(100, len(df)))
        for _, row in sample_df.iterrows():
            if pd.notna(row['image']):
                try:
                    images = ast.literal_eval(row['image'])
                    for img_path in images:
                        full_path = os.path.join('../../data/images/official_data_iccv_final', img_path)
                        if not os.path.exists(full_path):
                            missing_files += 1
                        total_files_checked += 1
                except:
                    pass
        print(f"\nFile existence check (sample of {len(sample_df)} rows):")
        print(f"  Checked {total_files_checked} images.")
        print(f"  Missing files: {missing_files}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_data_quality('../../data/raw/mimic_cxr_aug_validate.csv')
    check_data_quality('../../data/raw/mimic_cxr_aug_train.csv')
