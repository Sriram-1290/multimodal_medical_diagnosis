import pandas as pd
import ast
import os

df = pd.read_csv('mimic_cxr_aug_validate.csv')
missing_count = 0
for idx, row in df.iterrows():
    if missing_count >= 5:
        break
    images = ast.literal_eval(row['image'])
    for img in images:
        path = os.path.join('official_data_iccv_final', img)
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            print(f"List dir of parent {os.path.dirname(path)}:")
            if os.path.exists(os.path.dirname(path)):
                print(os.listdir(os.path.dirname(path)))
            else:
                print("Parent dir also missing!")
            missing_count += 1
            break
