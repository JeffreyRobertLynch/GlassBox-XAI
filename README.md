# GlassBox XAI – Lightweight Medical Imaging Prototype

A solo-built, end-to-end system for interpretable, infrastructure-light medical imaging. Designed to explore segmentation, explainability, and auditability under hardware and data constraints.

---

## Disclaimer

---

## Architecture & Training

- **Custom architecture**: U-Net with attention bottleneck  
- **Dataset**: ISIC 2018 – Task 1: Lesion Segmentation  
- **Custom Loss functions**: Dice, Tversky, Hybrid
- **Training Pipeline** includes robust augmentation, fine tuning, and callbacks for early stopping, iterative checkpoints, and LR scheduling 
- **Model-agnostic** utilities for evaluation, batch metrics, confusion matrices, comparison, and explainability
- **Fully modular** XAI + evaluation pipeline supports any model, batch, or layer combination
- **Modular** image processing stack supports image/mask alignment, augmentation, and pre-inference transformations
- **Plug-and-play** functions for metric reporting, variant comparison, and visual overlays 

---

## Metrics 

| Model              | Dice     | IoU      | Precision | Recall   | Pixel Accuracy | Global F1 Score |
|-------------------|----------|----------|-----------|----------|----------------|----------|
| **Dice-Optimized**   | **0.8751** | **0.8000** | **0.9028**  | 0.8291   | **0.9272**      | 0.8644   |
| **Balance-Optimized**| 0.8734   | 0.7925   | 0.8787    | 0.8564   | 0.9267         | **0.8674** |
| **Recall-Optimized** | 0.8573   | 0.7669   | 0.8280    | **0.8936** | 0.9182         | 0.8595   |

---

## Pixel-Level Error Tradeoffs by Model

| Model                     | False Positives (%) | False Negatives (%) |
|---------------------------|---------------------|----------------------|
| Dice-Optimized            | ~2.5%               | ~4.8%                |
| Balance-Optimized         | ~3.3%               | ~4.0%                |
| Recall-Optimized          | ~5.2%               | ~3.0%                |

---

## Model Comparison

Trained and evaluated three model variants with same architecture but different loss functions. Outputs compared over shared test batch.

**Note** A single batch featuring mean performance is displayed across all visuals for transparency. Matches closely to Dice-Optimized (Model 1) average performance on full 1000 image test set. Best case and worst case included in deep dive. 

<details>
<summary>View Comparative Model Outputs</summary>

![Multi-Model - Variant Comparison Visual](output/multi_model_batch_a_1.png)
*Side-by-side comparison of segmentation output vs. ground truth across three model variants.*

</details>

---

## Explainability (XAI)

Integrated multiple interpretability techniques to support expert review and regulatory workflows. Outputs generated at inference time for each test image.

- Layerwise Grad-CAM (decoder layers shown)  
- Integrated Gradients  
- Saliency Maps (logits & sigmoid-scaled)  
- Pixel Confidence Overlays  

<details>
<summary>View Grad-CAM Output</summary>

![Model 1 - Grad-CAM Decoder Layer Output](output/layer_dec_model_1_batch_a_1.png)
*Decoder-layer activation via Grad-CAM. Full end-to-end layerwise mapping available in deep dive.*

</details>

<details>
<summary>View Integrated Gradients</summary>

![Model 1 - Integrated Gradients Output](output/int_grad_model_1_batch_a_1.png) 
*Map using Integrated Gradients.*

</details>

<details>
<summary>View Saliency Map (Logits)</summary>

![Model 1 - Saliency Map Logits Output](output/sal_map_model_1_batch_a_1_raw.png) 
*Saliency based on raw logits.*

</details>

<details>
<summary>View Pixel Confidence Overlay</summary>

![Model 1 - Confidence Map Output](output/conf_map_model_1_batch_a_1.png)
*Overlay displaying class confidence for each pixel, grouped by like pixels.*

</details>

---

## Auditability & HITL Design

- Full Layerwise Grad-CAM, Integrated Gradients, Saliency, Pixel Confidence
- Variant model comparative segmentation
- Outputs and heatmap overlays designed for **review, regulatory compliance, and clinical workflow integration**  
- Transparency layers directly support **human-in-the-loop (HITL)** processes
- Full training, segmentation, and XAI pipeline runs on **modest hardware** with no external dependencies (cloud, expanded data, pretrained models)

---

## Generalizability

System structure, transparency methods, and audit frameworks generalize to:

- Healthcare diagnostics  
- Agriculture 
- Quality control 

---

### Contact
