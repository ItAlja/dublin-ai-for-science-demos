# 💬 Post-Session Q&A Add-On: How to Guide AlphaFold 3 With Custom Templates in Practice

**Event:** Google.org AI for Science Accelerator (Dublin, October 15, 2026) · Health & Life Sciences Track  
**Speaker:** Aleksandra Łucja Jaworska, AI/ML Customer Engineer, Google Cloud EMEA  
**Context:** *Added in response to a live question from the audience at the end of the session regarding Slide 17 ("How to Train Each Model"): **"Since AlphaFold 3 has fixed weights and cannot be fine-tuned with LoRA, how does guiding it with custom templates actually look in practice?"***

---

## 1. Why AlphaFold 3 Doesn't Use LoRA
* Unlike **MedGemma**, **TxGemma**, or **Cell2Sentence**—which are causal language models built on Gemma where LoRA (`r=8, alpha=16`) adapts the linear projection layers—**AlphaFold 3** is a geometric structure prediction model combining a **Pairformer** trunk with a **3D Denoising Diffusion module** over atomic coordinates (`x, y, z`).
* In the official [`google-deepmind/alphafold3`](https://github.com/google-deepmind/alphafold3) open-source release, DeepMind provides the **JAX inference pipeline (`run_alphafold.py`)** and frozen model weights.
* Instead of updating model weights via backpropagation, you steer AlphaFold 3 **at inference time** by supplying custom structural templates, custom sequence alignments, and chemical constraints inside its input JSON file (`fold_input.json`).

---

## 2. How Guiding AlphaFold 3 Works in Practice: `fold_input.json`

Every AlphaFold 3 job is driven by a `fold_input.json` file. By default, if you only pass a protein's amino-acid sequence, AlphaFold 3 runs a CPU data pipeline (`jackhmmer`, `nhmmer`, `hmmsearch`) to search public genetic databases for **Multiple Sequence Alignments (MSAs)** and search the public Protein Data Bank (PDB) for **3D templates**.

When your lab wants to steer AlphaFold 3 using your own experimental data (for example, forcing a kinase or receptor into an **open vs. closed** conformation, modeling a proprietary engineered protein, or docking a novel small molecule from **TxGemma**), you override three sections in **`fold_input.json`**:

### A. Custom 3D Structural Templates (`"templates"` — `.cif` / `.mmcif` files)
Suppose your lab has solved an X-ray crystal structure, a cryo-EM structure, or generated a molecular dynamics snapshot of your target protein in a specific state (`my_lab_open_conformation.cif`).
* In `fold_input.json`, under your protein chain, you add a `"templates"` list pointing to your `.cif` file (`"mmcifPath"`) along with a 0-based index mapping between your target sequence residues (`"queryIndices"`) and the template residues (`"templateIndices"`).
* **What happens under the hood:** AlphaFold 3's **Template Embedder** reads the 3D atomic coordinates from your `.cif` file, computes a **pairwise distance matrix (distogram)** between residues, and injects those geometric features into the Pairformer trunk before the diffusion module generates the 3D structure—physically pulling the predicted fold toward your custom template.

### B. Custom or Masked MSAs (`"unpairedMsaPath"` or `"unpairedMsa": ""`)
* **Custom `.a3m` Alignment:** If your lab has proprietary homolog sequences, you pass `"unpairedMsaPath": "/path/to/custom_alignment.a3m"`.
* **Turning Off Evolutionary Override (`"unpairedMsa": ""`):** A common practical challenge in structural biology is when thousands of public homologous sequences in the default MSA pull AlphaFold 3 toward the canonical "closed" state, overriding your custom 3D template. To force AlphaFold 3 to follow your custom 3D template strictly, researchers pass `"unpairedMsa": ""` (an empty MSA string) alongside their custom `"templates"`.

### C. Novel Drug Ligands (`"smiles"` / `"userCCD"`) & Covalent Bonds (`"bondedAtomPairs"`)
* When pairing **TxGemma** (1D virtual screening) with **AlphaFold 3** (3D pocket modeling), you pass your top candidate molecule's `SMILES` string directly under `"ligand"`.
* If the drug is a covalent inhibitor (such as Osimertinib binding covalently to `Cys797` in `EGFR`), you explicitly pin the covalent bond in `"bondedAtomPairs"` using `[entity_id, 1_based_residue_index, atom_name]`.

---

## 3. Complete Practical Example (`fold_input.json`)

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

---

## 4. Running AlphaFold 3 With Your Custom Template

Once your custom template and MSA are specified in `fold_input.json`, you run `run_alphafold.py` with `--run_data_pipeline=false` (which skips the slow CPU search over public databases and immediately runs GPU inference using your custom template):

```bash
python run_alphafold.py \
  --json_path=fold_input.json \
  --model_dir=/weights/af3 \
  --output_dir=./results \
  --run_data_pipeline=false
```

### Inspecting the Output Confidence
AlphaFold 3 writes the predicted 3D complex to `egfr_custom_open_pocket_with_drug_model.cif` and the confidence metrics to `summary_confidences.json`:
* **`pLDDT` (0–100):** Per-atom local confidence.
* **`PAE` (Predicted Aligned Error, in Å):** Confidence in the relative 3D orientation between domains or between the protein pocket and the drug ligand (lower Ångström error = darker green on the PAE plot = higher confidence).
* **`ipTM` (Interface predicted TM-score):** Confidence in the binding interface between chains/ligands (`>0.75` indicates a high-confidence complex).

---

## 🔗 References
* **Official AlphaFold 3 Input JSON Schema:** <https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md>
* **Official AlphaFold 3 Output Documentation:** <https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md>
* **Abramson et al. (*Nature*, 2024):** *"Accurate structure prediction of biomolecular interactions with AlphaFold 3"*
