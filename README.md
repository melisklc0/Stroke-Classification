# Stroke Classification System via Knowledge Distillation & Deep Learning

[![Live Demo](https://img.shields.io/badge/🤗%20Live_Demo-Try_it_now-blue?style=for-the-badge)](https://huggingface.co/spaces/melisklc0/stroke-classification)
[![Model](https://img.shields.io/badge/🤗%20Model-EfficientNet--B0-yellow)](https://huggingface.co/melisklc0/efficientnet-b0-stroke-distilled)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-3k%20folds-green)](https://huggingface.co/datasets/melisklc0/stroke-classification)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-green)](https://www.kaggle.com/datasets/melisklc0/stroke-classification-brain-ct)

**Project Type:** Capstone Project & Modular Data Science Portfolio  
📄 **[Read the Full Thesis Report (PDF) - TR](thesis/Stroke_Classification_Thesis_Report.pdf)**  
📖 **[Türkçe için tıklayın](README_TR.md)** — Turkish README (data layout, pipeline, results).

---

## 🎯 Try the live demo (no install)

**Upload a head CT scan and see Stroke vs. No-Stroke predictions in your browser — takes about 30 seconds.**

### 👉 [**Launch Live Demo on Hugging Face →**](https://huggingface.co/spaces/melisklc0/stroke-classification)

**Streamlit** interface backed by the distilled **EfficientNet-B0** student model (~98% F1 on our validation protocol). No local setup or GPU required.

> **Disclaimer:** Research and portfolio demo only — **not for clinical use** or medical decision-making.

---

## 📦 Published artifacts

Everything needed to reproduce, extend, or demo the project is hosted outside GitHub:

| Resource | What you get | Link |
| -------- | ------------ | ---- |
| **Live demo** | Streamlit app — upload CT, view probabilities | [Hugging Face Space](https://huggingface.co/spaces/melisklc0/stroke-classification) |
| **Model weights** | KD EfficientNet-B0 (`model.pth`, `load_model.py`) | [melisklc0/efficientnet-b0-stroke-distilled](https://huggingface.co/melisklc0/efficientnet-b0-stroke-distilled) |
| **Dataset** | 15k augmented folds + external test split (~3 GB) | [Hugging Face Dataset](https://huggingface.co/datasets/melisklc0/stroke-classification) |
| **Dataset (mirror)** | Same data on Kaggle | [Kaggle Dataset](https://www.kaggle.com/datasets/melisklc0/stroke-classification-brain-ct) |

---

## 🚀 Executive Summary

This project delivers a highly optimized, end-to-end Machine Learning pipeline designed to accelerate the diagnosis of strokes from CT scans. By synthesizing **Knowledge Distillation (KD)** and **Ensemble Learning**, we successfully compressed a computationally heavy state-of-the-art model into a lightweight, highly accurate diagnostic tool.

The system transitions traditional diagnostic delays into a rapid **Clinical Decision Support System**, demonstrating that lightweight student models can outperform their complex teachers while significantly reducing hardware (CPU/GPU) constraints.

Training code is in this repo; the dataset, distilled model, and Streamlit demo are linked in the table above.

## 🛠 Tech Stack & Methodologies

  * **Domain:** Computer Vision, Medical Imaging (CT Scans)
  * **Deep Learning Architecture:** PyTorch, Transfer Learning (ResNet, DenseNet, Inception, EfficientNet)
  * **Advanced Techniques:** Knowledge Distillation, Soft-Voting Ensemble Learning, Data Augmentation
  * **MLOps & Infrastructure:** `uv` (Fast Package Management), YAML-driven configuration, Modular Data Pipelines

## 👥 Authors

  * **Melis Kılıç**
  * **Esra Koç**

**Advisor:** Assoc. Prof. Dr. Kali Gürkahraman

## 📊 Key Achievements & Quantifiable Results

  * **Outperforming the Teacher:** The baseline Teacher model (InceptionV3) achieved a strong 97.6% F1 Score. Through Knowledge Distillation, the lightweight Student model (**EfficientNetB0**) surpassed its teacher, reaching a **98.0% F1 Score** with a fraction of the computational cost.
  * **Peak Ensemble Stability:** By applying a Soft-Voting strategy to just two distilled models (KD-EfficientNetB0 + B3), the system achieved a maximum accuracy of **98.2%**, reducing the overall error rate to a mere 1.7%.
  * **Robust Generalization:** Validated the model on a completely external Kaggle dataset. Despite differing scanner calibrations and color matrices, the system successfully maintained its ability to detect critical stroke patterns without catastrophic failure.

## 🔬 Architecture & Data Pipeline

### 1. Data Engineering & Preprocessing
* **Sources (raw):** 
  * Primary Dataset: [Open Data Portal of the Turkish Ministry of Health](https://acikveri.saglik.gov.tr/Home/DataSetDetail/1)
  * External Validation: [Kaggle Head CT Hemorrhage Dataset](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage)
* **Processed dataset:** 3-fold augmented splits (~15k images), published on Hugging Face and Kaggle (see table above).
* **Class Unification:** The raw dataset originally contained "No Stroke", "Bleeding", and "Ischemia" classes. To optimize for emergency triage, "Bleeding" and "Ischemia" were computationally merged into a unified **"Stroke"** class for binary classification.
* **Mitigating Overfitting:** Augmented the dataset to 15,000 normalized tensor images via rotations, zoom shifts, and horizontal flips to ensure robust feature extraction. Cross-validated using a 3-Fold CV strategy.

### 2. The Training Pipeline (3 Phases)

1.  **Baseline CNN Training:** 7 distinct CNN architectures were trained independently. InceptionV3 emerged as the optimal "Teacher".
2.  **Knowledge Distillation:** Transferred the "dark knowledge" from InceptionV3 to efficient student models by mimicking logit-level behaviors.
3.  **Ensemble Network:** Combined the probabilistic outputs of the best KD models using Soft-Voting to hedge against single-model bias.

## 💻 Installation & Modular Pipeline Usage Guide

The architecture is designed as a modular, reproducible Data Science Pipeline. The entire workflow is governed by `config.yaml` and executed sequentially via `run_pipeline.py`.

### 1. Environment Setup

We use the ultrafast `uv` package manager for dependency resolution.

#### Installing UV (If not installed)
*   **Windows:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
*   **macOS / Linux (Bash):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*(Note: You may need to restart your terminal after installation for the command to be recognized.)*

#### Virtual Environment & Installation
```bash
uv venv
uv pip install .
```

#### 📦 Dataset download

Log in to Hugging Face, then pull the published dataset (~3 GB) into `dataset/`:

```bash
uv run python -c "from huggingface_hub import login; token=input('HuggingFace Token: '); login(token.strip())"
uv run python run_pipeline.py --step pull_data
```

For inference without training, use the published model or Streamlit demo (table above).

### 2. Data Setup Configurations

The `--step data` command dynamically adapts to your local data structure. Choose the scenario that matches your raw data:

  * **Scenario A: You have Raw Images**  
  If you only have raw images, structure your `dataset/` directory like this:

    ```text
    dataset/
    ├── Stroke/       (All patient scans)
    └── No-Stroke/    (All healthy scans)
    ```

    Running `uv run python run_pipeline.py --step data` will automatically process, augment, and split these into `dataset/Fold1`, `dataset/Fold2`, etc.

  * **Scenario B: You have Pre-split Folds**  
  If your data is already split, place the folders directly under the `dataset/` directory:

    ```text
    dataset/
    ├── Fold1/
    ├── Fold2/
    └── Fold3/
    ```

    In this case, **you do not need to run the data step.** The system will auto-detect the hierarchy and is ready for training (`--step train_cnn`).

### 3. Executing the Pipeline Workflow (`run_pipeline.py`)

**Step 1: Clustering, Balancing, and Augmentation**  
Prepare the dataset based on your `config.yaml` settings (e.g., K-Fold count = 3).

```bash
uv run python run_pipeline.py --step data
```

**Step 2: Baseline CNN Training (The Teacher)**  
Select a model via `config.yaml` (e.g., `inceptionv3`) and train it to establish the baseline. *(Note: If using Inception, ensure `image_size: 299` is set in the config).*

```bash
uv run python run_pipeline.py --step train_cnn
```

**Step 3: Knowledge Distillation (KD) Training (The Student)**  
Provide the path to the trained InceptionV3 `.pth` weight and train a new student model (e.g., `efficientnetb0`). The loss function hyperparameters are optimized in the code at $\alpha=0.7$ and $\tau=5.0$.

```bash
uv run python run_pipeline.py --step train_kd
```

**Step 4: Soft-Voting Ensemble Tests**  
List the `.pth` paths of your top-performing KD Students under the `ensemble -> weights` section in `config.yaml`. Set your `validation_mode` (internal/external) and run:

```bash
uv run python run_pipeline.py --step ensemble
```

> **Visual Artifacts & Log Management:**
>
>   - **Metrics & Plots:** Automatically saved to `results/cnn/`, `results/kd/`, or `results/ensemble/`. This includes Confusion Matrices (Heatmaps), Training Loss curves, and full classification reports (F1, Recall) in both PNG and TXT formats.
>   - **Original Thesis Results:** Archived in the `thesis/results/` directory.
>   - **Model Weights (.pth):** Checkpoints are securely saved to `checkpoints/`