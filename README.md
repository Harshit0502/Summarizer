# Structure-Aware Long-Document Legal Judgment Summarization (IN-Abs) 

This project is a **notebook research pipeline** for summarizing **long Indian legal judgments** (IN-Abs dataset).  
It builds intermediate artifacts (index → embeddings/clusters → cleaned text → structured sections → token-aware chunks → TF‑IDF skeletons → LED-ready JSONL) and finally fine-tunes **LED** (`allenai/led-base-16384`) for abstractive summarization.

> **Note:** This repo/notebook is shared as a *code snapshot* for review. Data is not included.

---

## What’s inside

- `Legal_Summarization_Pipeline.ipynb` *(rename your notebook to this if you want)*  
  End-to-end phases:
  1. **Phase 0:** Build dataset index (cap train to **1500** cases)
  2. **Phase 1:** Sentence embeddings + **KMeans clustering (K=6)**
  3. **Phase 2:** Text cleaning + save cleaned judgments
  4. **Phase 3:** Discourse structuring into canonical sections
  5. **Phase 4:** Token-aware chunking for long inputs (LED tokenizer)
  6. **Phase 5:** TF‑IDF skeleton extraction aligned to gold summaries
  7. **Phase 6:** Build LED training JSONL (`input_text`, `target_text`) with truncation budget
  8. **Phase 7:** Fine-tune LED + ROUGE evaluation

---

## Dataset

The notebook expects an **IN-Abs** folder layout like:

```
IN-Abs/
  train-data/
    judgement/   <case_id>.txt
    summary/     <case_id>.txt
  test-data/
    judgement/   <case_id>.txt
    summary/     <case_id>.txt
```

In Kaggle, you set:

- `ROOT = "/kaggle/input/<your-dataset>/dataset/IN-Abs"`

> Data is **not bundled** in this share. Keep it private and mount it in Kaggle/Colab.

---

## Quickstart (recommended: Kaggle)

1) Upload/open the notebook in Kaggle (GPU recommended).  
2) Update dataset path in **Phase 0**:

```python
ROOT = "/kaggle/input/legal-datav2/dataset/IN-Abs"  # change this
```

3) Run cells **top-to-bottom**.

The notebook writes all artifacts to:

- `/kaggle/working/` *(writable in Kaggle)*

---

## Outputs (artifacts written by the notebook)

### Indexes
- `cases_index_indabs_1500.csv` (Phase 0)
- `cases_index_indabs_1500_phase1.csv` (adds `cluster_id`, `cluster_label`)
- `cases_index_indabs_1500_phase2.csv` (adds `cleaned_path`)
- `cases_index_indabs_1500_phase3.csv` (adds `structured_path`)
- `cases_index_indabs_1500_phase4.csv` (adds `chunked_path`, `num_chunks`)
- `cases_index_indabs_1500_phase5.csv` (adds `skeleton_path`, `num_skeleton_sentences`)

### Embeddings / clustering (Phase 1)
- `phase1_embeddings_train.npy`, `phase1_embeddings_test.npy`
- `phase1_train_case_ids.json`, `phase1_test_case_ids.json`

Embedding model:
- `sentence-transformers/all-MiniLM-L6-v2`

Clustering:
- `KMeans(n_clusters=6, random_state=42, n_init=10)`

### Cleaned judgments (Phase 2)
- `cleaned/<case_id>.txt`

### Structured judgments (Phase 3)
- `structured/<case_id>.json` with sections:
  - `FACTS`, `ISSUES`, `ARGUMENTS`, `REASONING`, `FINAL_ORDER`

### Chunked long inputs (Phase 4)
- `chunked/<case_id>.json` with token-aware chunks

Tokenizer:
- `allenai/led-base-16384`

Key settings:
- `MAX_SOURCE_TOKENS = 4096`
- `MAX_CHUNK_TOKENS = 3800`

### Skeleton extraction (Phase 5)
- `skeleton/<case_id>.json`

Method:
- TF‑IDF vectors for judgment sentences vs summary sentences
- cosine similarity
- selects candidate judgment sentences aligned to summary sentences

Key settings:
- `TOP_K_PER_SUMMARY = 3`
- `MIN_SIM_THRESHOLD = 0.10`
- `MAX_SKELETON_SENTENCES = 60`

### LED training data (Phase 6)
- `led_train.jsonl`, `led_val.jsonl`, `led_test.jsonl`

`input_text` structure (high level):
- **cluster token** (e.g., cluster id)
- **skeleton block** (optional highlight tags)
- **doc block** with `[SECTION=...]` tags

Truncation:
- source truncated with a **budget** to preserve cluster+skeleton as much as possible
- `MAX_SOURCE_TOKENS = 4096`, `MAX_TARGET_TOKENS = 512`

### Fine-tuned model (Phase 7)
- output directory: `./led_finetuned_indabs`

Training config (Phase 7 default):
- model: `allenai/led-base-16384`
- batch size: `1`
- gradient accumulation: `4` (effective batch ~4)
- epochs: `2`
- learning rate: `2e-5`
- warmup ratio: `0.03`
- weight decay: `0.01`
- fp16: `True`
- global attention: **first token** set to global attention mask
- evaluation: ROUGE (via `evaluate`)

---

## Requirements

The notebook installs/uses:

- `transformers`
- `datasets`
- `evaluate`
- `accelerate`
- `sentence-transformers`
- `torch`
- `numpy`, `pandas`
- `scikit-learn`
- `tqdm`

If you want a minimal `requirements.txt`:

```txt
transformers
datasets
evaluate
accelerate
sentence-transformers
torch
numpy
pandas
scikit-learn
tqdm
```

---

## Sharing checklist

Before sharing the notebook publicly:
- Remove any API keys/tokens and `.env` usage (none should be present).
- Replace private dataset paths with placeholders.
- Do **not** include dataset files or raw judgments in outputs.
- Clear cell outputs if they display sensitive text.

---
