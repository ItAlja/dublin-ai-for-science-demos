# 🧬 MedGemma & the Google Open Science Stack — Hands-On Colab Notebooks

**Event:** Google.org AI for Science Accelerator (Dublin, October 15, 2026) · Health & Life Sciences Track  
**Speaker:** Aleksandra Łucja Jaworska, AI/ML Customer Engineer, Google Cloud EMEA

Welcome! This repository contains the three executable **Google Colab notebooks** (with pre-run outputs included) accompanying the Dublin session **"MedGemma & the Open Science Stack"**, plus a **Post-Session Q&A Add-On** answering the live audience question on guiding **AlphaFold 3** with custom templates.

Click any **"Open in Colab"** badge below to launch the notebook directly in Google Colab on a single GPU (`T4` or `L4`).

---

## 🚀 Quick-Launch Notebooks (1-Click Open in Colab)

| # | Notebook & Architecture | What It Demonstrates | Launch in Google Colab |
| :--- | :--- | :--- | :--- |
| **1** | **`01_Brain_Tumor_MRI_LoRA_FineTuning.ipynb`**<br>*(Use Case 1 · 4-Bit QLoRA on Private Brain MRI)* | Downloads real Brain Tumor MRI scans (`Glioma`, `Meningioma`, `Pituitary`) from Hugging Face, reproduces the `33.3%` out-of-the-box baseline, trains a rank-8 LoRA adapter (`r=8, alpha=16`) with `-100` prompt masking (`bfloat16` on L4 or `NF4 + float32` on T4), and plots the **Before (`33.3%`) vs. After (`89.2%`) 3×3 Confusion Matrix**. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ItAlja/dublin-ai-for-science-demos/blob/main/01_Brain_Tumor_MRI_LoRA_FineTuning.ipynb) |
| **2** | **`02_MedGemma_TxGemma_Variant_and_Drug_Triage.ipynb`**<br>*(Use Case 2 · 3-Stage Verification Pipeline)* | Filters 1,201 candidate variants in deterministic Python, verifies wild-type protein sequences against live **UniProt** & **ClinVar** APIs, runs **MedGemma 1.5 4B** for mechanism synthesis, scores candidate therapies with **TxGemma** (`Erlotinib` vs. `Osimertinib`), and enforces a strict **JSON Schema + Citation Guardrail**. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ItAlja/dublin-ai-for-science-demos/blob/main/02_MedGemma_TxGemma_Variant_and_Drug_Triage.ipynb) |
| **3** | **`03_Combined_2in1_Master_Demo_MedGemma_and_TxGemma.ipynb`**<br>*(Combined End-to-End Master Workflow)* | Loads `google/medgemma-1.5-4b-it` onto the GPU **once** and runs both **Use Case 2 (Python Verification + TxGemma Drug Triage)** and **Use Case 1 (Brain Tumor MRI LoRA Fine-Tuning)** in a single end-to-end pipeline. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ItAlja/dublin-ai-for-science-demos/blob/main/03_Combined_2in1_Master_Demo_MedGemma_and_TxGemma.ipynb) |
| **Q&A** | **[`04_QA_AddOn_AlphaFold3_Custom_Templates.md`](./04_QA_AddOn_AlphaFold3_Custom_Templates.md)**<br>*(Post-Session Q&A Add-On · AlphaFold 3)* | Practical guide & `fold_input.json` code example answering the live audience question from Dublin: **"How does guiding AlphaFold 3 with custom 3D templates and MSAs work in practice without LoRA?"** | [View Q&A Guide](./04_QA_AddOn_AlphaFold3_Custom_Templates.md) |

---

## ⚙️ Prerequisites to Run in Google Colab
1. In Google Colab, select **Runtime → Change runtime type → T4 GPU or L4 GPU**.
2. Accept the open-weight model license on Hugging Face for [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it) and add your Hugging Face token (`HF_TOKEN`) to Colab Secrets (the key icon in the left sidebar).

---

## 💬 Post-Session Q&A Add-On: Guiding AlphaFold 3 With Custom Templates in Practice

> **Note (Added following live audience Q&A in Dublin on October 15, 2026):**  
> *At the end of the session, an attendee asked a great question regarding Slide 17: **"Since AlphaFold 3 has fixed weights and cannot be fine-tuned with LoRA, how does guiding it with custom templates actually look in practice?"** Here is how structural biology and drug discovery teams do this in practice.*

### 1. Why You Don't Use LoRA on AlphaFold 3
* Unlike **MedGemma**, **TxGemma**, or **Cell2Sentence** (which are autoregressive Gemma language models where LoRA adapts linear projection layers), **AlphaFold 3** combines a **Pairformer** trunk with a **3D Denoising Diffusion module** that directly predicts 3D atomic coordinates (`x, y, z`).
* In the official [`google-deepmind/alphafold3`](https://github.com/google-deepmind/alphafold3) release, DeepMind provides the **JAX inference pipeline (`run_alphafold.py`)** and frozen model weights. Rather than updating weights via backpropagation, you steer the 3D diffusion model **at inference time** by injecting your lab's structural and evolutionary constraints into its input JSON file (`fold_input.json`).

### 2. How You Guide AlphaFold 3 in Practice (`fold_input.json`)
By default, when you pass a raw protein sequence to AlphaFold 3, it searches public genetic databases to build a **Multiple Sequence Alignment (MSA)** and searches the public Protein Data Bank (PDB) for **3D templates**.

To adapt AlphaFold 3 to your lab's proprietary structures or force a specific protein conformation (e.g., an *open* vs. *closed* binding pocket, or a custom engineered protein scaffold), you override three fields inside **`fold_input.json`**:

1. **Custom 3D Structural Templates (`"templates"` — `.cif` / `.mmcif` files):**
   * You supply your own experimental crystal structure, cryo-EM structure, or MD snapshot via `"mmcifPath"` along with a 0-based residue mapping (`"queryIndices"` and `"templateIndices"`).
   * **What happens under the hood:** AF3's **Template Embedder** converts the 3D atomic coordinates from your `.cif` file into a **pairwise distance matrix (distogram)** between residues and feeds it into the Pairformer trunk before diffusion runs—physically pulling the predicted 3D fold toward your template's conformation.
2. **Custom or Masked MSAs (`"unpairedMsaPath"` or `"unpairedMsa": ""`):**
   * You can pass your lab's proprietary sequence alignment (`"unpairedMsaPath": "/path/to/custom.a3m"`).
   * **Practical conformational trick:** If public evolutionary homologs in the MSA are pulling AF3 toward the wrong "default" state and overriding your custom 3D template, structural biologists pass an **empty MSA (`"unpairedMsa": ""`)** alongside `"templates"` so AF3 relies strictly on your custom 3D template!
3. **Custom Ligands (`"smiles"` / `"userCCD"`) & Covalent Bonds (`"bondedAtomPairs"`):**
   * For novel small molecules (for example, top hits screened by **TxGemma**), you pass the molecule's `SMILES` string and optionally lock covalent bonds between specific protein atoms and ligand atoms (such as `EGFR Cys797` bonding to a covalent inhibitor).

### 3. Concrete Example: `fold_input.json` With a Custom Lab Template, Custom MSA & Covalent Drug

```json
{
  "name": "egfr_custom_open_pocket_with_drug",
  "dialect": "alphafold3",
  "version": 2,
  "modelSeeds": [1, 2, 3],
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVL...",
        "unpairedMsaPath": "/lab_data/custom_egfr_alignment.a3m",
        "templates": [
          {
            "mmcifPath": "/lab_data/my_lab_open_conformation.cif",
            "queryIndices":    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "templateIndices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
          }
        ]
      }
    },
    {
      "ligand": {
        "id": "B",
        "smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
      }
    }
  ],
  "bondedAtomPairs": [
    [
      ["A", 797, "SG"],
      ["B", 1, "C12"]
    ]
  ]
}
```

### 4. Running Inference With Your Custom Template (Skipping the Public DB Search)
When you already provide your own `"templates"` and `"unpairedMsaPath"` (or `"unpairedMsa": ""`), you run `run_alphafold.py` with `--run_data_pipeline=false` so it skips the slow CPU database search and runs the GPU diffusion model directly on your custom inputs:

```bash
python run_alphafold.py \
  --json_path=fold_input.json \
  --model_dir=/weights/af3 \
  --output_dir=./results \
  --run_data_pipeline=false
```

* **Official AlphaFold 3 Input Documentation:** [`google-deepmind/alphafold3/docs/input.md`](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)
