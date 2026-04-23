# 🏥 Multimodal Medical Diagnosis — System Architecture

> **Document Version**: 1.0  
> **Last Updated**: April 24, 2026  
> **Scope**: End-to-end architecture covering data pipeline, model internals, training loop, inference engine, API server, and web frontend.

---

## Table of Contents

1. [High-Level System Overview](#1-high-level-system-overview)
2. [Data Pipeline Architecture](#2-data-pipeline-architecture)
3. [Model Architecture — Deep Dive](#3-model-architecture--deep-dive)
4. [Training Pipeline Architecture](#4-training-pipeline-architecture)
5. [Inference & Generation Engine](#5-inference--generation-engine)
6. [Serving Architecture (API + Frontend)](#6-serving-architecture-api--frontend)
7. [Directory & Module Map](#7-directory--module-map)
8. [Tensor Flow & Shape Analysis](#8-tensor-flow--shape-analysis)
9. [Design Decisions & Trade-off Review](#9-design-decisions--trade-off-review)
10. [Security & Reliability Considerations](#10-security--reliability-considerations)

---

## 1. High-Level System Overview

The system is an **image-to-text medical AI pipeline** that ingests multi-view Chest X-Ray (CXR) images and produces structured radiology reports. It is a full-stack application spanning from raw data processing to a production-ready web dashboard.

```mermaid
graph TB
    subgraph "Data Layer"
        RAW["Raw MIMIC-CXR-AUG CSVs"]
        IMG["CXR Image Store"]
        CLEAN["Cleaned CSVs"]
        RAW -->|cleanup_datasets.py| CLEAN
    end

    subgraph "Model Layer"
        VE["Vision Encoder<br/>DenseNet-121"]
        PROJ["Linear Projection<br/>1024 → 768"]
        TD["Text Decoder<br/>Bio_ClinicalBERT"]
        VE --> PROJ --> TD
    end

    subgraph "Training Layer"
        DS["MedicalReportDataset<br/>PyTorch DataLoader"]
        TRAIN["Training Loop<br/>Mixed Precision + Grad Accum"]
        CKPT["Checkpoint<br/>best_model.pth"]
        TB["TensorBoard Logs"]
        CLEAN --> DS
        IMG --> DS
        DS --> TRAIN
        TRAIN --> CKPT
        TRAIN --> TB
    end

    subgraph "Serving Layer"
        API["FastAPI Server<br/>uvicorn"]
        FE["Web Frontend<br/>HTML/CSS/JS"]
        API -->|Static Mount| FE
        CKPT -->|Load on Startup| API
    end

    USER["Radiologist / User"] -->|Upload CXR Views| FE
    FE -->|POST /api/predict| API
    API -->|JSON Report| FE
```

### System Summary

| Aspect | Technology |
|:---|:---|
| **Vision Backbone** | DenseNet-121 (ImageNet pretrained) |
| **Language Backbone** | Bio_ClinicalBERT (MIMIC-III pretrained) |
| **Fusion Strategy** | Cross-Attention via projected visual tokens |
| **Generation** | Beam Search with Repetition Penalty |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS (Glassmorphism UI) |
| **Training** | PyTorch AMP + Gradient Accumulation |
| **Monitoring** | TensorBoard |

---

## 2. Data Pipeline Architecture

### 2.1 Raw Data Ingestion

The project consumes the **MIMIC-CXR-AUG** dataset, a structured version of the MIMIC-CXR database augmented for multimodal training.

```mermaid
graph LR
    subgraph "Raw Sources"
        CSV_T["mimic_cxr_aug_train.csv"]
        CSV_V["mimic_cxr_aug_validate.csv"]
        IMGS["data/images/official_data_iccv_final/"]
    end

    subgraph "ETL Pipeline"
        A1["Drop Unnamed columns"]
        A2["Parse stringified lists<br/>ast.literal_eval"]
        A3["Clean 'nan' strings<br/>from view arrays"]
        A4["Verify every image<br/>exists on disk"]
    end

    subgraph "Clean Output"
        OUT_T["mimic_cxr_aug_train_cleaned.csv"]
        OUT_V["mimic_cxr_aug_validate_cleaned.csv"]
    end

    CSV_T --> A1 --> A2 --> A3 --> A4 --> OUT_T
    CSV_V --> A1
    A4 --> OUT_V
```

### 2.2 CSV Schema

| Column | Type | Description |
|:---|:---|:---|
| `subject_id` | int | Unique patient identifier |
| `image` | List[str] | Relative paths to all CXR images for the study |
| `view` | List[str] | Anatomical view labels (`'PA'`, `'AP'`, `'LATERAL'`) |
| `AP` / `PA` / `Lateral` | List[str] | Pre-filtered view-specific filenames |
| `text` | List[str] | Original radiology reports (Findings & Impression) |
| `text_augment` | List[str] | Synthetically augmented report variants |

### 2.3 Dataset Class (`MedicalReportDataset`)

The PyTorch `Dataset` handles the complex mapping from tabular CSV + image files to batched tensors.

**Key Responsibilities:**
1. **Multi-view loading**: Reads up to 3 images per study (AP, PA, Lateral)
2. **Zero-padding**: Missing views are filled with `torch.zeros(3, 224, 224)` to maintain batch shape consistency
3. **Image transforms**: Resize → ToTensor → ImageNet Normalize
4. **Tokenization**: Reports tokenized with Bio_ClinicalBERT tokenizer, padded/truncated to `max_length`

**Output per sample:**
```
{
    'image':          Tensor(3, 3, 224, 224),   # NumViews × C × H × W
    'input_ids':      Tensor(max_length),        # Token IDs
    'attention_mask':  Tensor(max_length)          # Padding mask
}
```

---

## 3. Model Architecture — Deep Dive

The model (`MedicalReportGenerator`) is an **encoder-decoder** architecture with three distinct stages.

### 3.1 Full Model Diagram

```mermaid
graph TB
    subgraph "Input"
        I["CXR Images<br/>(B, 3, 3, 224, 224)"]
    end

    subgraph "Stage 1: Vision Encoder"
        FLAT["Flatten Views<br/>(B×3, 3, 224, 224)"]
        DN["DenseNet-121<br/>features block"]
        SPAT["Spatial Features<br/>(B×3, 1024, 7, 7)"]
        RESHAPE["Reshape to Sequence<br/>(B×3, 49, 1024)"]
        AGG["Multi-View Aggregation<br/>(B, 147, 1024)"]
    end

    subgraph "Stage 2: Feature Projection"
        LINEAR["nn.Linear(1024, 768)<br/>Visual Projection"]
        PROJ_OUT["Projected Features<br/>(B, 147, 768)"]
    end

    subgraph "Stage 3: Text Decoder"
        TOK["Input Token IDs<br/>(B, seq_len)"]
        BERT["BertLMHeadModel<br/>is_decoder=True<br/>add_cross_attention=True"]
        LOGITS["Output Logits<br/>(B, seq_len, 28996)"]
    end

    I --> FLAT --> DN --> SPAT --> RESHAPE --> AGG
    AGG --> LINEAR --> PROJ_OUT
    PROJ_OUT -->|encoder_hidden_states| BERT
    TOK -->|input_ids| BERT
    BERT --> LOGITS
```

### 3.2 Vision Encoder — `CXRVisionEncoder`

**File**: `scripts/models/vision_encoder.py`

| Property | Detail |
|:---|:---|
| **Backbone** | `torchvision.models.densenet121` |
| **Pretrained Weights** | ImageNet (DenseNet121_Weights.DEFAULT) |
| **Layers Used** | `densenet.features` only (classifier head removed) |
| **Output Channels** | 1024 |
| **Spatial Grid** | 7×7 (for 224×224 input) → 49 patches |
| **Freezing** | Optional via `freeze_weights` parameter |

**How it works:**

DenseNet-121 processes each image independently. The final feature map is a 7×7 spatial grid with 1024 channels. Instead of global average pooling (which would discard spatial information), the grid is **flattened into a sequence of 49 visual tokens**, each carrying 1024-dimensional features. This preserves spatial locality for the decoder's cross-attention.

```
Input:  (B, 3, 224, 224)  →  DenseNet Features  →  (B, 1024, 7, 7)
                                                  →  reshape  →  (B, 49, 1024)
```

### 3.3 Multi-View Aggregation

**File**: `scripts/models/multimodal_generator.py` → `encode_images()`

This is the critical fusion step. Instead of processing views independently and averaging, the system **concatenates spatial tokens** from all views:

```python
# Input: (B, NumViews=3, C=3, H=224, W=224)
flat_images = images.view(-1, C, H, W)           # (B×3, 3, 224, 224)
flat_features = vision_encoder(flat_images)        # (B×3, 49, 1024)
view_features = flat_features.view(B, 3, 49, 1024) # (B, 3, 49, 1024)
aggregated = view_features.reshape(B, 147, 1024)   # (B, 147, 1024)
```

**Result**: The decoder receives **147 visual tokens** (3 views × 49 patches each). During cross-attention, each generated word can attend to any spatial region across **all** views simultaneously.

### 3.4 Visual Projection

A single `nn.Linear(1024, 768)` layer maps DenseNet's 1024-dim features to Bio_ClinicalBERT's expected 768-dim hidden size. This is necessary because the cross-attention mechanism in BERT requires matching dimensions.

### 3.5 Text Decoder — `RadiologyReportDecoder`

**File**: `scripts/models/text_decoder.py`

| Property | Detail |
|:---|:---|
| **Base Model** | `emilyalsentzer/Bio_ClinicalBERT` |
| **HF Class** | `BertLMHeadModel` |
| **Config Mods** | `is_decoder=True`, `add_cross_attention=True`, `tie_word_embeddings=False` |
| **Vocab Size** | 28,996 tokens |
| **Hidden Size** | 768 |
| **Num Layers** | 12 Transformer layers |
| **Max Length** | Configurable (default 128, training uses 512) |

**How it works:**

Bio_ClinicalBERT is normally a bidirectional encoder. By setting `is_decoder=True`, it applies **causal masking** (each token can only attend to previous tokens). With `add_cross_attention=True`, each Transformer layer gains an additional **cross-attention sub-layer** that attends to the visual features provided as `encoder_hidden_states`.

```mermaid
graph LR
    subgraph "Each Decoder Layer"
        SA["Self-Attention<br/>(Causal Masked)"]
        CA["Cross-Attention<br/>(to Visual Tokens)"]
        FFN["Feed-Forward<br/>Network"]
        SA --> CA --> FFN
    end
```

**Why Bio_ClinicalBERT?** It is pre-trained on ~2M clinical notes from the MIMIC-III database, giving it strong prior knowledge of medical terminology, abbreviations, and report structure.

---

## 4. Training Pipeline Architecture

**File**: `scripts/training/train.py`

### 4.1 Training Loop Diagram

```mermaid
graph TB
    subgraph "Initialization"
        CFG["Config: batch=4, epochs=20,<br/>accum_steps=4, max_len=512"]
        MODEL["MedicalReportGenerator<br/>→ GPU"]
        OPT["AdamW with Differential LR"]
        RESUME["Resume from Checkpoint<br/>(if exists)"]
    end

    subgraph "Training Epoch"
        LOAD["DataLoader<br/>(shuffle=True, workers=4)"]
        AMP["AMP Autocast<br/>(FP16 Forward)"]
        LOSS["CrossEntropyLoss<br/>(ignore padding)"]
        BACK["Scaled Backward"]
        ACCUM{"Step % 4 == 0?"}
        STEP["Optimizer Step +<br/>Scaler Update"]
        LOG["TensorBoard Log<br/>(every 10 steps)"]
    end

    subgraph "Validation"
        VLOAD["Val DataLoader"]
        VLOSS["Compute Val Loss"]
        CHECK{"val_loss < best?"}
        SAVE["Save Full Checkpoint"]
    end

    CFG --> MODEL --> OPT --> RESUME
    RESUME --> LOAD --> AMP --> LOSS --> BACK --> ACCUM
    ACCUM -->|Yes| STEP --> LOG
    ACCUM -->|No| LOAD
    LOG --> VLOAD --> VLOSS --> CHECK
    CHECK -->|Yes| SAVE
```

### 4.2 Key Training Features

#### Differential Learning Rates
The model uses **two parameter groups** with different learning rates:

| Parameter Group | LR | Rationale |
|:---|:---|:---|
| Pretrained (DenseNet + BERT core) | `1e-5` | Preserve pre-trained knowledge |
| New (cross-attention + projection) | `4e-5` | Faster adaptation for new layers |

#### Gradient Accumulation
With `accumulation_steps=4` and `batch_size=4`, the effective batch size is **16**. This allows training with multi-view inputs on limited VRAM.

#### Mixed Precision Training
Uses `torch.amp.autocast('cuda')` + `GradScaler` for FP16 forward/backward passes, reducing memory usage and speeding up training.

#### Full State Checkpointing
The checkpoint saves **all state** needed for seamless resumption:
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss
}
```

#### Teacher Forcing
During training, the model receives the **ground truth token sequence** shifted by one position:
- **Input**: `tokens[:, :-1]` (all tokens except last)
- **Target**: `tokens[:, 1:]` (all tokens except first)

This is standard autoregressive training with `CrossEntropyLoss`.

---

## 5. Inference & Generation Engine

**File**: `scripts/models/multimodal_generator.py` → `generate()`

### 5.1 Beam Search Algorithm

```mermaid
graph TB
    START["Start: CLS token"] --> INIT["Initialize K beams<br/>score=0.0"]
    INIT --> LOOP{"Max length<br/>reached?"}
    LOOP -->|No| EXPAND["For each beam:<br/>Get logits → Top-K tokens"]
    EXPAND --> PENALTY["Apply Repetition<br/>Penalty to seen tokens"]
    PENALTY --> SCORE["Score = parent + log_prob"]
    SCORE --> PRUNE["Keep Top-K beams<br/>globally"]
    PRUNE --> DONE{"All beams<br/>hit SEP?"}
    DONE -->|No| LOOP
    DONE -->|Yes| BEST["Return highest<br/>scoring beam"]
    LOOP -->|Yes| BEST
```

### 5.2 Generation Parameters

| Parameter | Default | Description |
|:---|:---|:---|
| `k` (beam width) | 5 | Number of parallel hypotheses |
| `max_length` | 128 | Maximum tokens to generate |
| `repetition_penalty` | 1.5 (train) / 2.0 (serve) | Penalty for repeating tokens |

### 5.3 Repetition Penalty Mechanism

To prevent degenerate outputs like `": : : : :"`, the generation applies a multiplicative penalty to already-generated tokens:

```python
for token_id in generated_tokens:
    if logits[token_id] > 0:
        logits[token_id] /= repetition_penalty   # reduce positive logits
    else:
        logits[token_id] *= repetition_penalty    # push negative logits further down
```

---

## 6. Serving Architecture (API + Frontend)

### 6.1 Backend — FastAPI

**File**: `scripts/app.py`

```mermaid
graph LR
    subgraph "FastAPI Server (port 8000)"
        STARTUP["Startup:<br/>load_resources()"]
        STATUS["/api/status<br/>GET"]
        PREDICT["/api/predict<br/>POST"]
        STATIC["/ (Static Files)<br/>index.html"]
    end

    subgraph "Model Runtime"
        CKPT["best_model.pth"]
        TOK["Bio_ClinicalBERT<br/>Tokenizer"]
        MDL["MedicalReportGenerator<br/>(eval mode)"]
    end

    STARTUP --> CKPT --> MDL
    STARTUP --> TOK
    PREDICT -->|"Transform Images<br/>→ (1,3,3,224,224)"| MDL
    MDL -->|"Beam Search"| PREDICT
```

#### API Endpoints

| Endpoint | Method | Purpose |
|:---|:---|:---|
| `/api/status` | GET | Health check — reports device, checkpoint status |
| `/api/predict` | POST | Accepts up to 3 image uploads (ap_view, pa_view, lateral_view), returns generated report |
| `/` | GET | Serves the static frontend (index.html) |

#### Image Processing Pipeline (Server-Side)
```
Upload → PIL.Image.open() → RGB convert → Resize(224×224) → ToTensor → ImageNet Normalize
→ Stack 3 views → Unsqueeze batch dim → (1, 3, 3, 224, 224) → model.generate()
```

Missing views are filled with `torch.zeros(3, 224, 224)`.

### 6.2 Frontend — Web Dashboard

**Files**: `scripts/static/index.html`, `script.js`, `style.css`

```mermaid
graph TB
    subgraph "UI Components"
        HEADER["Header<br/>Logo + Status Badge"]
        HERO["Hero Section<br/>Title + Description"]
        UPLOAD["Upload Card<br/>3 Drop Zones + Folder Upload"]
        RESULT["Result Card<br/>4 States"]
        FOOTER["Footer<br/>Tech Stack"]
    end

    subgraph "Result States"
        S1["Welcome State<br/>(default)"]
        S2["Loading State<br/>(pulse animation)"]
        S3["Result State<br/>(Findings + Impression)"]
        S4["Error State<br/>(error message)"]
    end

    subgraph "User Flow"
        U1["Check /api/status"] --> U2["Upload Images"]
        U2 --> U3["Click Generate"]
        U3 --> U4["POST /api/predict"]
        U4 --> U5["Display Report<br/>(typewriter effect)"]
    end

    UPLOAD --> RESULT
    RESULT --> S1
    RESULT --> S2
    RESULT --> S3
    RESULT --> S4
```

#### Frontend Features
- **Glassmorphism UI**: Frosted-glass card effects with `backdrop-filter: blur(12px)`
- **Animated Background**: Three floating gradient blobs with CSS keyframe animations
- **Smart Folder Upload**: Auto-maps files named `ap.jpg`/`pa.jpg`/`lateral.jpg` to correct slots
- **Typewriter Effect**: Generated findings are displayed character-by-character (15ms/char)
- **Responsive Grid**: Two-column layout collapses to single column at 1024px

---

## 7. Directory & Module Map

```
multimodal_medical_diagnosis/
├── data/
│   ├── raw/                          # Original MIMIC-CXR-AUG CSVs
│   ├── processed/                    # Cleaned CSVs (output of ETL)
│   ├── images/official_data_iccv_final/  # CXR image store
│   └── infer_ease/                   # Pre-organized inference samples
├── models/
│   ├── checkpoints/best_model.pth    # Trained model weights
│   └── logs/                         # TensorBoard event files
├── scripts/
│   ├── models/                       # Neural network definitions
│   │   ├── vision_encoder.py         # DenseNet-121 feature extractor
│   │   ├── text_decoder.py           # Bio_ClinicalBERT decoder
│   │   └── multimodal_generator.py   # Fusion model + beam search
│   ├── data_prep/                    # Data ETL pipeline
│   │   ├── dataset.py                # PyTorch Dataset class
│   │   ├── cleanup_datasets.py       # CSV cleaning & image verification
│   │   ├── analyze_datasets.py       # Data quality auditing
│   │   └── debug_missing.py          # Missing image debugger
│   ├── training/
│   │   ├── train.py                  # Full training loop
│   │   └── inference.py              # Standalone inference test
│   ├── static/                       # Web frontend
│   │   ├── index.html                # Dashboard HTML
│   │   ├── script.js                 # Client-side logic
│   │   └── style.css                 # Glassmorphism styles
│   ├── app.py                        # FastAPI server
│   ├── setup_hf.py                   # Pre-cache HuggingFace models
│   ├── prepare_infer_ease.py         # Prepare inference test samples
│   ├── analyze_logs.py               # TensorBoard log analyzer
│   ├── diagnose_features.py          # Feature variation diagnostics
│   ├── diagnose_inference.py         # Inference difference test
│   └── diagnose_logits.py            # Logit variation diagnostics
└── tasks/                            # Project management
```

### Module Dependency Graph

```mermaid
graph BT
    VE["vision_encoder.py"] --> MG["multimodal_generator.py"]
    TD["text_decoder.py"] --> MG
    MG --> TRAIN["train.py"]
    MG --> APP["app.py"]
    MG --> INFER["inference.py"]
    DS["dataset.py"] --> TRAIN
    MG --> DIAG1["diagnose_features.py"]
    MG --> DIAG2["diagnose_inference.py"]
    MG --> DIAG3["diagnose_logits.py"]
```

---

## 8. Tensor Flow & Shape Analysis

This section traces the exact tensor shapes through every stage of the pipeline.

### 8.1 Forward Pass (Training)

```
Step                          Shape                    Notes
─────────────────────────────────────────────────────────────────
Input Images                  (B, 3, 3, 224, 224)      B=batch, 3 views, RGB, 224×224
Flatten Views                 (B×3, 3, 224, 224)       Treat each view independently
DenseNet Features             (B×3, 1024, 7, 7)        1024 channels, 7×7 spatial
Reshape to Sequence           (B×3, 49, 1024)          49 patches per view
Reshape by Batch              (B, 3, 49, 1024)         Group back by batch
Concatenate Views             (B, 147, 1024)           3×49 = 147 visual tokens
Visual Projection             (B, 147, 768)            Match BERT hidden size
Input IDs (shifted)           (B, seq_len-1)           Teacher-forced tokens
BERT Decoder Output           (B, seq_len-1, 28996)    Logits over vocabulary
Targets (shifted)             (B, seq_len-1)           Ground truth next tokens
CrossEntropy Loss             scalar                   Ignoring pad_token_id
```

### 8.2 Inference Pass (Beam Search)

```
Step                          Shape                    Notes
─────────────────────────────────────────────────────────────────
Input Images                  (1, 3, 3, 224, 224)      Single study
Visual Encoding               (1, 147, 768)            Same as training
Initial Beam                  (1, 1)                   Just [CLS] token
Per Step: Decoder             (1, t, 28996)            t = current length
Next Token Logits             (28996,)                 Last position logits
Apply Rep. Penalty            (28996,)                 Penalize seen tokens
Top-K Selection               K candidates             Expand each beam
Prune to K Beams              K beams kept              Best scores survive
Final Output                  string                   Decoded best beam
```

---

## 9. Design Decisions & Trade-off Review

### 9.1 Architecture Choices

| Decision | Choice Made | Alternatives | Rationale |
|:---|:---|:---|:---|
| **Vision backbone** | DenseNet-121 | ResNet-50, ViT, EfficientNet | DenseNet's dense connections are well-suited for medical imaging; feature reuse preserves fine details |
| **Spatial tokens vs. global pool** | Spatial (49 tokens/view) | Global average pooling | Spatial tokens let the decoder attend to specific image regions — critical for localized findings |
| **Multi-view fusion** | Concatenation | Average pooling, attention-based | Concatenation preserves all spatial info; lets cross-attention learn view relevance implicitly |
| **Language model** | Bio_ClinicalBERT | GPT-2, T5, BioGPT | Pre-trained on MIMIC-III clinical notes — strong domain match for radiology language |
| **BERT as decoder** | Causal mask + cross-attn | Native decoder model | Leverages clinical pre-training; native decoders lack medical domain knowledge |
| **Generation strategy** | Beam Search (k=5) | Greedy, Nucleus Sampling | Beam search produces more coherent reports; repetition penalty prevents loops |
| **Projection layer** | Single linear layer | MLP, attention adapter | Simplicity; a linear projection is sufficient for dimension matching |

### 9.2 Training Strategy Review

| Decision | Choice | Rationale |
|:---|:---|:---|
| **Differential LR** | 1e-5 pretrained / 4e-5 new | Prevents catastrophic forgetting of ImageNet + clinical knowledge |
| **Gradient accumulation** | 4 steps (eff. batch=16) | Multi-view inputs are VRAM-intensive; accumulation simulates larger batches |
| **Mixed precision** | FP16 via AMP | ~2× speedup, ~40% memory reduction with minimal accuracy loss |
| **Checkpointing** | Full state (model + optimizer + epoch + best_loss) | Enables seamless training resumption without loss spikes |
| **Max sequence length** | 512 tokens | Radiology reports can be lengthy; 128 truncates important findings |

### 9.3 Strengths

1. **Multi-view awareness**: 147 visual tokens across 3 views gives the decoder comprehensive spatial context
2. **Domain-specific language**: Bio_ClinicalBERT's pre-training on clinical notes produces natural medical language
3. **Robust data pipeline**: Image existence verification prevents silent training errors from missing files
4. **Production-ready serving**: FastAPI + static frontend is deployable with a single command
5. **Diagnostic tooling**: Three separate diagnostic scripts help debug feature collapse, logit variation, and inference quality

### 9.4 Potential Improvements

| Area | Current Limitation | Possible Enhancement |
|:---|:---|:---|
| **View encoding** | All views share one DenseNet | View-specific encoders or view-type embeddings |
| **Zero-padding** | Missing views → zero tensors | Learned mask tokens or attention masking |
| **Projection** | Single linear layer | Multi-layer adapter with residual connections |
| **Generation** | Basic beam search | Nucleus sampling, length normalization, n-gram blocking |
| **Evaluation** | Only val loss | BLEU, ROUGE, CheXpert F1, clinical accuracy metrics |
| **Attention vis** | Not implemented | Grad-CAM heatmaps on CXR regions for explainability |

---

## 10. Security & Reliability Considerations

### 10.1 Model Loading Safety
- Checkpoint loading uses `map_location=device` to prevent CUDA↔CPU mismatches
- Legacy checkpoint format (raw state_dict) is gracefully handled alongside full-state format
- Failed optimizer restoration falls back to fresh optimizer without crashing

### 10.2 API Robustness
- `/api/status` provides health checks before allowing inference
- Missing checkpoint → HTTP 503 with clear error message (not a crash)
- Image processing errors are caught with full traceback logging
- CORS middleware enabled for flexible frontend deployment

### 10.3 Data Integrity
- `cleanup_datasets.py` verifies every image path on disk before including rows
- Dataset class wraps individual image loads in try/except with zero-tensor fallback
- `analyze_datasets.py` provides pre-training data quality auditing

### 10.4 Known Environment Issues
- **OpenMP conflict**: `KMP_DUPLICATE_LIB_OK=TRUE` set at script top to prevent `libiomp5md.dll` crashes on Windows
- **HuggingFace symlinks**: Warning suppressed via `HF_HUB_DISABLE_SYMLINKS_WARNING=1`

---

> **End of Architecture Document**  
> For getting started instructions, see [README.md](file:///d:/projects/multimodal_medical_diagnosis/README.md).
