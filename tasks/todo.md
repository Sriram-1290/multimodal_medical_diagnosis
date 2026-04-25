# Dataset Analysis Plan

## 1. Plan Mode Default
- [x] Inspect the root directory structure and files.
- [x] Inspect the structure of `official_data_iccv_final` and determine what it contains.
- [x] Analyze the schema and contents of `mimic_cxr_aug_train.csv` and `mimic_cxr_aug_validate.csv`.
- [x] Investigate the contents of the `minic-crx-dataset.zip` if needed (list files inside).
- [x] Check if there's any context from the previous conversation `Analyze Project Directory` that adds specific requirements.
- [x] Compile a comprehensive analysis report on the datasets.

## 2. Execution
- [x] Run scripts or commands to peek at the CSV files.
- [x] Summarize the columns, data types, and any noticeable patterns in the CSVs.
- [x] Walk through the `official_data_iccv_final/files` directory.
- [x] Create an artifact `dataset_analysis.md` with the full detailed report.

## 3. Data Cleanup
- [x] Drop redundant columns (`Unnamed: 0`, `Unnamed: 0.1`) from both train and validate CSVs.
- [x] Parse stringified lists (`['abc', 'def']`) back into proper Python lists using `ast.literal_eval`.
- [x] Remove `'nan'` strings from the parsed `view` arrays.
- [x] Verify image existence in `official_data_iccv_final/files/` and filter out rows with missing images.
- [x] Save the cleaned data to `mimic_cxr_aug_train_cleaned.csv` and `mimic_cxr_aug_validate_cleaned.csv`.
- [x] Run a verification check on the new cleaned datasets.

## 4. Project Organization
- [x] Create typical ML subdirectories (`data/raw`, `data/processed`, `data/images`, `scripts/data_prep`).
- [x] Move raw CSVs and dataset zip to `data/raw`.
- [x] Move cleaned CSVs to `data/processed`.
- [x] Move `official_data_iccv_final` images to `data/images`.
- [x] Move utility python scripts to `scripts/data_prep`.
- [x] Update dataset pathing in scripts to reflect the new directory structure.
- [x] Create model structure directories (`scripts/models`, `scripts/training`, `models/checkpoints`, `models/logs`).

## 5. Model Architecture Implementation
- [x] Draft a comprehensive multimodal architecture plan for Image-to-Text generation.
- [x] Incorporate user feedback on the architecture design.
- [x] Implement Vision Encoder (`vision_encoder.py`).
- [x] Implement Language Decoder (`text_decoder.py`).
- [x] Implement Multimodal Generator integration (`multimodal_generator.py`).

## 6. Data Loading & Training Pipeline
- [x] Create `scripts/data_prep/dataset.py` to handle multimodal loading (Images + Reports).
- [x] Implement `scripts/training/train.py` with the training loop, loss calculation, and checkpointing.
- [x] Implement utility for report generation (greedy/beam search).

## 7. Project Ready
- [x] Project architecture and pipeline verified. Training ready.

## 8. Phase 2: Refinements
- [x] Implement Beam Search in `MedicalReportGenerator.generate()`.
- [x] Implement View Aggregation (aggregrating AP/PA/Lateral views).
- [x] Add TensorBoard logging to `train.py`.


## 9. Bug Fixes
- [x] Resolve `OMP: Error #15` (Duplicate OpenMP runtime) in `train.py`.
- [x] Install missing `tensorboard` dependency.

## 10. Training Script Enhancements
- [x] Add model parameter counting (total and trainable) to `train.py`.
