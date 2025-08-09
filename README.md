# GlassBox-XAI
Attention U-Net for Medical Image Segmentation with XAI Suite  

---

## Data Ethics & Full Disclaimer

**This project is for research and demonstration purposes only. It is not a medical device and is not intended for clinical use, diagnosis, or treatment.**

**All medical images shown are sourced from the ISIC 2018 Challenge Dataset. A publicly released, fully anonymized dataset curated for non-commercial research. No private health information (PHI) is included.**

**All results and visual outputs are generated deep learning models and evaluated using the official ISIC 2018 test set for accuracy metrics. This work has not been validated in clinical settings.**

**No claims are made regarding diagnostic accuracy, safety, or fitness for medical decision-making.**

**Visualizations (e.g., segmentation masks, saliency maps, Grad-CAM overlays) are interpretability tools designed to support human understanding of model behavior. They are not clinical indicators.**

This work reflects a commitment to transparency, explainability, and responsible AI development in medical imaging research. Please refer to the [ISIC 2018 dataset license and terms of use](https://challenge2018.isic-archive.com/) for attribution and usage guidelines.

---

## Problem Domain
This project focuses on **automated binary segmentation of skin lesions** in medical images, a fundamental step toward early melanoma detection. Unlike basic classification, which simply detects the presence of a lesion, accurate segmentation outlines precise lesion boundaries. Potential applications include:

- **Risk Triage:** Flagging high-risk areas for deeper evaluation.

- **Quantitative Tracking:** Monitoring lesion growth or regression over time.

- **Decision Support:** Integration with Computer-Aided Diagnosis (CAD) tools and Human-in-the-Loop (HITL) Systems.

### Why Explainability (XAI) Is Critical in Healthcare AI
Transparency in model decisions is essential for the safe and ethical adoption of AI in healthcare. XAI enables:

- **Clinician Trust & Human Oversight:** Visual tools like saliency maps and Grad‑CAM help clinicians verify model reasoning at the pixel level, supporting HITL workflows and shared decision-making.

- **Regulatory & Ethical Compliance:** XAI enables audit trails, bias detection, and decision transparency increasingly required by medical regulators (e.g., FDA).

- **Model Debugging & Iterative Improvement** Visualizations reveal model behavior under edge cases, guiding Subject Matter Expert (SME) feedback loops during development and evaluation.

- **Real-World Utility:** Studies show interpretable tools are more likely to gain clinician trust and improve diagnostic performance when used as assistive systems [(Holzinger et al., 2017)](https://arxiv.org/abs/1712.09923).

---

## Solution Overview

### Project Goals

- Build a fully custom solution for **ISIC 2018: Task 1 – Binary Segmentation** with performance that matches or exceeds top 2018 leaderboard entries.
- Adhere to ISIC 2018 challenge dataset splits (Training, Validation, Test) for quantitative benchmarking.
- Impose real-world constraints to reflect clinical deployment environments:
  - **Local-only pipeline**: Train, test, and deploy entirely on-device to ensure **PHI security**, **reproducibility**, and **offline deployability**.
  - **No pretrained models**: Supports **full model ownership**, traceability, and end-to-end auditability.
  - **No ensemble or high-cost methods**: Keeps **resource demands low**, enabling **deployment on modest hardware**.
  - **No cloud or external services**: Prevents data leakage and supports **secure, private use cases**.
  - **ISIC 2018-only training data**: Reflects **real-world limitations** in curated, domain-specific datasets.
- Train **three specialized model variants** to explore trade-offs across **precision**, **recall**, and **balanced performance**.
- Integrate a modular **XAI toolkit** to support **transparency**, **human-in-the-loop workflows**, and **regulatory readiness**.

---

### Resources

- **Dataset**: Only the **ISIC 2018 Task 1 dataset** was used. This set is fully anonymized and publicly released for research. The dataset contains ~3000 dermoscopic images with corresponding ground truth segmentation masks, curated for the ISIC 2018 challenge. High-quality and widely used for benchmarking.
- **Hardware**: Local workstation with **NVIDIA RTX 3080**, **32 GB RAM**, and **AMD 3700X CPU**. No cloud or distributed compute used.
- **Timeline**: ~160 solo development hours over 4 weeks (not including documentation).
- **Development Context**: Developed by a solo researcher with specialization in **AI in Healthcare (Stanford University)**.

---

### Performance Metrics

Model performance is measured by comparing its predicted segmentations to expert-annotated ground truth masks from the test set. These test images were **not seen during training**, so they reveal how well the model can **generalize** to new, unseen cases. This is a key indicator of real-world utility.

Each metric captures a different aspect of segmentation quality:

- **Dice Coefficient:** Measures how well the predicted lesion area overlaps with the actual lesion. More tolerant of small boundary errors in large lesions but highly sensitive to errors in small lesions, where precision matters most. Favored in medical imaging because it aligns better with clinical priorities, where missing a small lesion can have greater consequences than imprecise edges on a large one.
- **Intersection over Union (IoU):** Similar to Dice but stricter. Penalizes all boundary mismatches equally, regardless of lesion size. Often used as a secondary reference metric.
- **Recall (Sensitivity):** Measures how well the model captures all true lesion areas. High recall reduces false negatives.
- **Precision:** Measures how many of the predicted lesion areas were actually correct. High precision reduces false positives.
- **F1 Score:** The harmonic mean of precision and recall. Useful when both over-detection and under-detection carry risk.
- **Pixel Accuracy:** Shows the overall percentage of correctly labeled pixels. Can be misleading in medical imaging, where lesions often occupy only a small part of the image. For example, if just 10% of an image contains a lesion and the model misses it entirely, it still scores 90% accuracy.

**All metrics inform model performance, but Dice is emphasized due to its closer alignment with clinical relevance and its widespread use in medical image segmentation research.**
---

### Variant Models

All three models use the **same core architecture**, but were trained and fine-tuned with **different loss functions** (Dice, Tversky, Weighted Hybrid, etc.) to optimize for distinct clinical and deployment goals:

- **Model 1 – Dice-Optimized:** Prioritizes high overall segmentation accuracy and maximization of the Dice Score.
- **Model 2 – Balance-Optimized:** Seeks an even trade-off between false positives and false negatives.
- **Model 3 – Recall-Optimized:** Minimizes false negatives, favoring sensitivity over specificity.

---

### Core Segmentation Features

- **Binary Segmentation Output:** The raw model output. A binary mask showing which pixels the model identifies as lesion and which it identifies as non-lesion.
- **Segmentation Overlay:** Predicted mask, or boundary decision, dimmed and laid over the original image for segmentation. An essential step for evaluating segmentation accuracy.  
- **Image Preprocessing Pipeline:** Fifteen toggleable preprocesing techniques, such as brightness adjustment and/or dilation, can be applied to images prior to segmentation to improve model performance. These techniques have been omitted for this demo. All visuals and reported metrics reflect performance on the unaltered, baseline dataset.
---

### Comparative Model Evaluation Features

- **Variant Comparison Segmentation Overlays:** Side-by-side visual comparison of all three model variants using the same image input. Highlights how each model handles boundary decisions differently based on specialization.
- **Test Set Performance Metrics:** Full evaluation across Dice, IoU, Precision, Recall, Pixel Accuracy, and F1 Score across all models.
- **Confusion Matrices:** Pixel-level false positive, false negative, true positive, and true negative rates for the entire test set across all models. High-level picture of strengths and trade-offs.

---

### Interpretability & XAI Features

- **Confidence Map Overlay:** Heatmap showing model confidence in its decisions across different regions of an image. Essential context for model decision transparency.
- **Saliency Map Overlay:** Highlights which pixels most strongly affect the model’s output, based on local sensitivity. Typically emphasizes edges or boundaries where predictions shift sharply. 
- **Integrated Gradients Overlay:** Measures cumulative influence of each pixel by comparing the image to a baseline. Unlike saliency maps, IG reveals focus on both the core and edges of a lesion, offering a more complete view of what drives model decisions.
- **Grad-CAM Visualizations**: Visualization of the progression from image input to model output, across all convolutional layers. Divided into encoder, attention, decoder, and output stages to show how segmentation decisions evolve through the network architecture.

---

### Model Metrics

| Model              | Dice     | IoU      | Precision | Recall   | Pixel Accuracy | F1 Score |
|-------------------|----------|----------|-----------|----------|----------------|----------|
| **Dice-Optimized**   | **0.8751** | **0.8000** | **0.9028**  | 0.8291   | **0.9272**      | 0.8644   |
| **Balance-Optimized**| 0.8734   | 0.7925   | 0.8787    | 0.8564   | 0.9267         | **0.8674** |
| **Recall-Optimized** | 0.8573   | 0.7669   | 0.8280    | **0.8936** | 0.9182         | 0.8595   |

---

### Solution Summary

GlassBox XAI achieves **Dice 0.8751** and **IoU 0.8000**, meeting or exceeding top entries from the ISIC 2018 leaderboard, **without relying on pretrained models, external data, or ensemble methods**. While some modern models (2024–2025) may report Dice scores above 0.90, these typically require computationally expensive techniques, substantial cloud infrastructure, or proprietary data pipelines.

By contrast, GlassBox was developed **entirely from scratch**, trained on limited data, and runs **securely on local hardware** with **end-to-end auditability**. It is suitable for real-world clinical deployment where **privacy, traceability, and model ownership** are critical design factors.

In short, this system delivers **high-tier performance under realistic clinical constraints**, while also offering **powerful visual diagnostics**, **exceptional model transparency**, and **flexibile, local deployability** not often found in more opaque, compute-heavy solutions.

These trade-offs, and potential development paths for GlassBox XAI, are explored more thoroughly in the **Future Work** section.

---

## Core Segmentation Features

The visuals below demonstrate the core segmentation capabilities of GlassBox XAI. While all features support all model variants and image batches, we primarily showcase **Model 1 (Dice-Optimized)** with **Batch A (Average Performance)** for clarity, consistency, and realistic baseline performance.

- **Batch A (Average Metrics):** Mirrors the models' average performance on the full test set. Represent typical, expected outcomes.
- **Batch B (High Metrics):** Outperforms full test set metrics. Illustatrates optimal use cases.
- **Batch C (Low Metrics):** Underperforms full test set metrics. Useful for evaluating edge cases or failure points.

---

#### Basic Segmentation Output - Model 1 - Batch A

This section demonstrates how raw model predictions are turned into clear visuals that support decision-making. The primary output is a binary segmentation mask, highlighting lesion regions on a per-pixel basis. Here we have the base image to be segmented, the expert annotated mask, and the models' outputted mask for comparison. 

Ground truth masks, expert-annotated and included for evaluation, allow us to compare model output visually and calculate performance metrics across the batch. 

![Model 1 - Segmentation Output Visual - Batch A](output/base_model_1_batch_a_1.png)

---

#### Segmentation Overlay - Model 1 - Batch A

To make predictions visually intuitive, we overlay the dimmed segmentation mask on the original image. This creates a human-readable output that can be compared directly with expert annotations. We include overlays for both the model and expert masks, along with batch-level performance metrics for context.

This visual uses Batch A, representing expected mean performance. The model's segmentation boundaries align closely with the expert-annotated boundaries, aside from images 4 and 5, which are undersegmented. Image 6 is slightly oversegmented, but still shows excellent shape alignment. 

![Model 1 - Segmentation Overlay Metric - Batch A](output/metric_overlay_model_1_batch_a_1.png)

![Model 1 - Segmentation Overlay Visual - Batch A](output/overlay_model_1_batch_a_1.png)

---

#### Segmentation Overlay - Model 1 - Batch B
This visual uses Batch B, representing optimal use case. The model's segmentation boundaries very closely mirror the expert-annotated boundaries. Only slight misalignment in images 3, 5, and 6.

![Model 1 - Segmentation Overlay Metric - Batch B](output/metric_overlay_model_1_batch_b_1.png)

![Model 1 - Segmentation Overlay Visual - Batch B](output/overlay_model_1_batch_b_1.png)

---

#### Segmentation Overlay - Model 1 - Batch C
This visual uses Batch C, representing edge cases and failure points. The model's segmentation boundaries align relatively closely with the expert-annotated boundaries for images 6, 7, and 8. However, the model is missing significant portions of the lesion in images 1, 2, 3, 4 and 5.

![Model 1 - Segmentation Overlay Metric - Batch C](output/metric_overlay_model_1_batch_c_1.png)

![Model 1 - Segmentation Overlay Visual - Batch C](output/overlay_model_1_batch_c_1.png)

---

## Comparative Model Evaluation
Building trust in AI systems requires more than clean outputs; it requires **objective, transparent evaluation**. This section compares the performance of three models trained on the same task to show how different training strategies affect generalization, error profiles, and suitability for specific deployment needs.

---

### Variant Comparison Segmentation Overlays - All Models - Batch A
Here we visualize the segmentation decision of all 3 variant models for comparison. For Batch A, differences are notable but not pronounced between variants. Other batches show greater divergence. 

![Multi-Model - Variant Comparison Metrics](output/metric_multi_model_batch_a_1.png)

![Multi-Model - Variant Comparison Visual](output/multi_model_batch_a_1.png)

---

### Model Performance Evaluation on Test Set - All Models - Test Set
All three models achieve strong overall performance on the held-out test set. However, their error patterns differ meaningfully. The clearest differentiator is how each model balances Precision and Recall, or the rate of false positives versus false negatives at the pixel level.

---

#### Model 1 - Dice-Optimized
Precision is higher; Recall is lower.

![Model 1 - Test Set Metrics](output/eval_model_1_1.png)

---

#### Model 2 - Balance-Optimized
Precision and Recall are balanced.

![Model 2 - Test Set Metrics](output/eval_model_2_1.png)

---

#### Model 3 - Recall-Optimized
Precision is lower; Recall is higher.

![Model 3 - Test Set Metrics](output/eval_model_3_1.png)

---

### Confusion Matrices - All Models - Test Set

Confusion matrices break down the models' correct and incorrect predictions on a per-pixel basis across the entire test set. This provides a more detailed view of where and how errors occur, especially for medical applications where every pixel may carry diagnostic weight.

These matrices reveal a consistent pattern across variants: as false positives increase, false negatives decrease, and vice versa. This does not indicate a drop in overall performance, each model remains highly accurate. Rather, it reflects how each was intentionally optimized for different risk profiles and deployment scenarios using custom loss functions. 

In medical imaging, false positives are more acceptable than false negatives if missing a condition, or part of a condition, could lead to serious consequences. In other contexts, such as when treatment carries significant risk or the condition is less severe, a higher false negative rate may be tolerable. Alternatively, we can aim for balance. These preferences can also be fine-tuned more granularly through thresholding and curve analysis to further specialize model behavior.

---

#### Model 1 - Dice-Optimized
~2.5% of test pixels are false positives; ~4.8% are false negatives. 

![Model 1 - Confusion Matrix](output/cm_model_1_1.png)

---

#### Model 2 - Balance-Optimized 
~3.3% of test pixels are false positives; ~4.0% are false negatives.

![Model 2 - Confusion Matrix](output/cm_model_2_1.png)

---

#### Model 3 - Recall-Optimized 
~5.2% of test pixels are false positives; ~3.0% are false negatives. 

![Model 3 - Confusion Matrix](output/cm_model_3_1.png)

---

## Interpretability & XAI
Understanding how and why a model makes its predictions is essential for trust, regulatory compliance, clinical safety, and real-world deployment.

This section introduces a suite of interpretability tools designed to help researchers, clinicians, and auditors explain, interrogate, and validate model behavior beyond raw performance metrics. These include:

- Pixel-level confidence maps

- Saliency visualizations (raw and sigmoid-scaled)

- Integrated gradients for attribution

- Full Grad-CAM layer-wise progression from input to output

Each method highlights a different aspect of the model's internal decision-making, allowing us to ask critical questions:

- Did the model focus on the lesion or background noise?

- Did it attend to misleading non-lesion features (e.g, hairs, moles, shadows) but ultimately learn to discard them correctly?

- Is it relying on human-visible features or irrelevant correlations?

- Does confidence behave as expected? High confidence in lesion cores but decreasing at the boundaries, as we expect to see, where the model has to decide precisely where to draw the segmentation boundary? 

- Does confidence drop near errors? Or does the model make errors with confidence?

- Are attention mechanisms effective? Do we see increased focus on relevant features and decreased focus on irrelevant features when applied?

Whether auditing model attention, evaluating edge cases, or designing a transparent system for actionable human-in-the-loop (HITL) decision support, **XAI is foundational**.

---

### Confidence Map Overlay - Model 1 - Batch A
This visualization shows the model’s segmentation prediction overlaid with a heatmap of per-pixel confidence, grouped into superpixels (clusters of visually similar pixels). Each region is color-coded based on how confidently the model believes it is part of the lesion (positive class).

---

#### What This Tells Us

- **Deep red lesion center:** High confidence in core lesion area.

- **Orange-yellow-green-blue borders:** Varying confidence near lesion edges (as expected).

- **Clear blue background:** High confidence that surrounding skin is not part of the lesion.

This behavior is exactly what we expect. High confidence in the lesion core, more uncertainty at boundaries, and strong rejection of background regions for all images. 

But pay particular attention to the difference between images 4, 5, and 6 compared to the others. When we examined the metrics earlier, we saw images 4 and 5 were undersegmented compared to the expert annotation. Image 6 was slightly oversegmented, but still had excellent shape alignment.  

---

#### Why It Matters
- **Pinpoints areas of uncertainty** to help human reviewers identify where predictions may need manual verification.

- **Supports HITL** workflows through auditing, especially in borderline or ambiguous cases.

- **Debugging & QA** during development iterations to detect patterns, edge cases, and failure modes like overconfidence in incorrect segmentaton decisions.

---

#### Technical Note
Superpixels are generated using the SLIC algorithm to divide the image into super-pixel regions. Confidence values are calculated by averaging the model’s probability outputs within each region, resulting in a smoother and more interpretable heatmap.

---

#### Real-World Use Case
This overlay could help a dermatologist triage ambiguous cases. Low-confidence boundaries may prompt further review, while high-confidence regions offer reassurance. This supports risk-aware decision-making and calibrates trust in model outputs on a per-image, or even per-region, basis.

---

#### Disclaimer
This system is for research and educational use only. Visuals and insights are based on the ISIC 2018 dataset and are not clinically validated.

---

![Model 1 - Confidence Map Output](output/conf_map_model_1_batch_a_1.png)

---

### Saliency Map Overlay - Raw Logits - Model 1 - Batch A

---

This visualization highlights where the model is most sensitive to small changes in the input image. Each pixel is color-coded based on the gradient magnitude flowing from the raw logits (pre-sigmoid output) with respect to the input. The result is a high-resolution "sensitivity heatmap" showing all areas the model reacts to whether helpful, distracting, or irrelevant. For example, we notice that non-lesion objects, like hairs, are consistently hot pixels here but these do not affect or confuse the model's final output.

---

#### What This Tells Us
- **Clusters of “hot” pixels:** Regions the model is highly sensitive to, often aligned with textures or segmentation boundaries.

- **Sensitivity outside the lesion:** Includes background features the model reacts to, such as lighting patterns or skin texture.

- **Wide attribution scope:** Captures all contributing input signals, not just the ones the model ultimately trusts in its final prediction.

This gives us a raw view of influence without filtering. Valuable for understanding early-stage model behavior or unwanted signal dependencies. Under close examination of noisy heatmaps, we can see that the hottest pixels generally correspond to segmentation boundaries. This will become clearer when we look at sigmoid-scaled saliency maps. 

Again, pay particular attention to images 4, 5, and 6 compared to the others. Images 4 and 5 were undersegmented. Image 6 was slightly oversegmented.  

---

#### Why It Matters
- **Exposes early model behavior** during R&D and debugging phases to identify spurious correlations (e.g., hair, shadows, borders).

- **Improves trust & safety** by revealing if a model is overly sensitive to irrelevant signals, overshadowing or distorting more relevant signals.

- **Enables data-driven fixes** as “hot spots” in irrelevant areas may signal a need for improved data augmentation, normalization, pre-processing techniques, or annotation quality.

---

#### Technical Note
Gradients are computed using TensorFlow’s GradientTape, capturing how small input changes influence the raw logits. This is achieved by cloning the model and temporarily removing the sigmoid activation from the output layer during backpropagation, allowing direct access to the raw logits. This method can appear noisy and doesn't necessarily represent final prediction logic. This is addressed in sigmoid-scaled saliency and integrated gradient techniques.

---

#### Real-World Use Case
In a research workflow, an ML engineer or clinical auditor might use this map to detect whether the model is reacting to imaging artifacts or unintended features. For example, consistent sensitivity to non-lesion areas could inform dataset refinement, retraining with augmentation, or model regularization. Additionally, we can use this to determine the effectiveness of attention mechanisms. Does a noisy image here ultimately lead to a clear, accurate, segmentation boundary?

---

#### Disclaimer
For research and educational use only. Visuals and insights are based on the ISIC 2018 dataset and are not clinically validated.

---

![Model 1 - Saliency Map Logits Output](output/sal_map_model_1_batch_a_1_raw.png)

---

### Saliency Map Overlay - Sigmoid-Scaled - Model 1 - Batch A

---

This visualization is much less noisy than the previous saliency map because the only hotspots are pixels that most influence the model’s final segmentation decision. This is computed using gradients of the sigmoid-activated output with respect to the input image. Unlike raw saliency maps which reflect all sensitivity, this map focuses only on what most affects the model’s final probability. The result is cleaner, more focused, and ideally emphasizes the lesion boundary.

---

#### What This Tells Us
- **Lesion boundary:** A thin, high-intensity band of pixels outlines the model’s segmentation decision.

- **Interior & background:** Remain mostly blue (low gradient), showing low influence on the final outcome.

Optimally, the model is not reacting broadly to irrelevant textures or lighting but is instead making sharp, boundary-based decisions. We see this clearly in the majority of images, where segmentation quality was excellent. 

However, compare images 4 and 5 to the majority. A much different pattern than the majority. A clustering of hot pixels versus clear segmentation outlines. 

Image 6, which was slightly oversegmented but still strongly aligned in shape, looks fairly clean here. More comparable to the majority than images 4 and 5. 

---

#### Why It Matters
- **Improves interpretability** by aligning attribution with the model’s actual output (post-activation), not just internal activations.

- **Highlights decisive features** rather than noisy sensitivities, making it easier for clinicians or auditors to validate the reasoning.

- **Strong HITL value** in expert review settings. Boundary-focused saliency supports targeted oversight of lesion delineation quality, particularly when compared and contrasted with raw logit saliency maps for the same input image.

---

#### Technical Note
This saliency map is computed using gradients derived from the sigmoid output, not raw logits. This constrains gradients to the final prediction layer, filtering out noisy activations. Compared to raw saliency, this produces smoother, cleaner, and more decision-aligned visualizations. Sigmoid-scaled saliency limits attention to the model’s final confidence output. Raw saliency highlights all input regions that influence internal activations, resulting in broader, noisier maps that often include background or texture noise. This difference is especially useful for auditing attention mechanisms and boundary fidelity when used together. Raw saliency shows sensitivity, sigmoid saliency shows decisions.

---

#### Real-World Use Case
In a dermatology AI pipeline, this overlay could help clinicians visually confirm the quality of lesion boundaries in low-confidence or borderline cases. If the focus aligns with clinically relevant features, trust is reinforced. If not, it flags potential model failure or edge-case behavior. Together, raw and sigmoid-scaled saliency offer a full-spectrum view of model behavior. Raw saliency reveals what the model notices, while sigmoid-scaled saliency shows what it actually acts on. This duality is critical for auditing feature dependence and evaluating attention mechanisms applied at intermediate convolutional layers.

---
 
#### Disclaimer
For research and educational use only. Visuals and insights are based on the ISIC 2018 dataset and are not clinically validated.

---

![Model 1 - Saliency Map Sigmoid Output](output/sal_map_model_1_batch_a_1.png)

---

### Integrated Gradients Overlay - Model 1 - Batch A

This visualization highlights the cumulative influence of each pixel on the model’s final segmentation decision, computed by tracing a path from a baseline (blank) image to the actual input. The result is a smoother, more stable attribution map compared to instantaneous saliency techniques. Integrated Gradients don’t just show what the model is sensitive to; they reveal which input features actually caused the prediction, traced along a path from blank input to final decision.

---

#### What This Tells Us
- **Bright, saturated regions:** These pixels consistently contributed to the model’s prediction as the image was progressively revealed and should correspond to the lesion.

- **Dark or desaturated regions:** These pixels had little to no effect on the model’s decision, suggesting they were ignored or discounted.

This reflects not just sensitivity, but causality. The full set of input regions the model relied on to reach its final output. Unlike the raw logits saliency maps, hot spots here do not align with misleading features, like hairs. 

Most images are remarkably consistent here, hot spots are correlated with the lesion. Image 4 is diffuse and a clear outlier, however. Most likely due to the presence of multiple differently colored lesion areas in one image. An edge case that can be understood and addressed through input preprocessing, expanding train data, or additional data augmentation.   

---

#### Why It Matters
- **Reduces noise** compared to raw saliency methods, offering more stable attribution maps.

- **Highlights causal contributions** instead of just attention or sensitivity, supporting model explainability during audits.

- **Improves interpretability in edge-cases** where gradients alone may be misleading or noisy.

Integrated gradients, alongside raw and sigmoid saliency, distinguish between what the model reacts to, what it acts on, and what it ultimately bases its decision on.

---

#### Technical Note
Like the raw logits saliency map, IG is applied to the model’s pre-sigmoid logits to ensure faithful attribution. Integrated gradients computes an attribution map by interpolating between a baseline image (blank) and the input image, calculating gradients at each step and summing them along the path. Fulfills theoretical axioms like sensitivity and implementation invariance, making it one of the most principled attribution methods in XAI literature. 

---

#### Real-World Use Case
In a dermatology AI pipeline, Integrated Gradients can help validate that a model is grounded in the right lesion features when raw saliency is too noisy or unclear. It can also reveal subtle dependencies on shape, texture, or contrast that affect generalization across skin tones, lighting conditions, or imaging devices. When used alongside saliency and confidence maps, model behavior can be interpreted through three complementary lenses: sensitivity, decision confidence, and causal attribution.


---

#### Disclaimer
This visualization is for research and educational purposes only. Visuals and insights are based on the ISIC 2018 dataset and are not clinically validated.

---

![Model 1 - Integrated Gradients Output](output/int_grad_model_1_batch_a_1.png)

---

#### Grad-CAM: Encoder Layers - Model 1 - Batch A

![Model 1 - Grad-CAM Encoder Layer Output](output/layer_enc_model_1_batch_a_1.png)

---

#### Grad-CAM: Attention Layers - Model 1 - Batch A

![Model 1 - Grad-CAM Attention Layer Output](output/layer_att_model_1_batch_a_1.png)

---

#### Grad-CAM: Decoder Layers - Model 1 - Batch A

![Model 1 - Grad-CAM Decoder Layer Output](output/layer_dec_model_1_batch_a_1.png)

---

#### Grad-CAM: Final Layer with Output - Model 1 - Batch A

![Model 1 - Grad-CAM Output Layer Output](output/layer_out_model_1_batch_a_1.png)

---

## Data
This project uses the publicly available, anonymized dataset from ISIC 2018: Task 1 - Lesion Segmentation, a globally recognized benchmark for skin lesion segmentation. See "Credits & References" section for direct links to the International Skin Image Collaboration (ISIC) website. 

- **High-Quality Curation:** The dataset was contributed by a consortium of international dermatology clinics and academic institutions. It includes dermoscopic images collected under expert supervision, making it a clinically relevant and representative dataset.

- **Anonymized & PHI-Free:** All images are de-identified and publicly released under ISIC’s data use policy. No patient-identifiable information or Protected Health Information (PHI) is present in the dataset.

- **Dataset Composition:** ~2,600 images and their corresponding binary lesion masks. Includes a wide range of skin tones, lighting conditions, lesion types, and occlusions (e.g., hair, ruler marks).

- **Split Integrity Maintained:** The original Training, Validation, and Test splits were strictly preserved. No data leakage between splits. This ensures that reported performance can be directly compared to past and future solutions.

- **Testing & Metrics:** During evaluation, the models' predicted segmentation masks are directly compared to the ground truth segmentation masks created by domain experts. All reported metrics — such as Dice Score, IoU, and Pixel Accuracy — are computed from this pixel-level comparison on the official test set.

- **Why This Matters:** High performance on this benchmark — without pretrained models or external datasets — suggests strong model generalization and real-world potential, even under resource-constrained, on-device conditions. The ISIC dataset continues to serve as a standard in dermatological AI research, and results on this set remain a meaningful indicator of segmentation quality, clinical alignment, and benchmarking rigor.

---

## Models & Metrics

### Architecture
GlassBox XAI uses a custom-built deep learning model based on U-Net, a specialized neural network architecture designed for precise image segmentation. Unlike typical convolutional networks that focus on classification, U-Net is shaped like a “U” and learns to both understand the big picture and recover fine details. It is ideal for identifying lesion boundaries pixel by pixel.

To further improve accuracy, especially in challenging or noisy images, **attention mechanisms** were added. These help the model focus on the most relevant regions in the image, such as lesion edges, while ignoring background distractions like hair or shadows.

- All models use the same custom U-Net architecture with added attention mechanisms.
- Each was trained from scratch on only ISIC 2018 data, no pretrained models.
- Regularization techniques including layer normalization and dropout were applied to reduce overfitting.
- Each variant was fine-tuned using different loss functions: Dice Loss, Tversky Loss, or Hybrid Dice-Tversky Loss.

---

### Model Performance Comparison


| Model        	 | Dice     | IoU      | Precision| Recall   |Pixel Accuracy | F1 Score |
|----------------|----------|----------|----------|----------|----------|----------|
| Dice-Optimized | **0.8751**   | **0.8000**   | **0.9028**   | 0.8291   | **0.9272**   | 0.8644   |
| Balance-Optimized | 0.8734   | 0.7925   | 0.8787   | 0.8564   | 0.9267   | **0.8674**   |
| Recall-Optimized | 0.8573   | 0.7669   | 0.8280   | **0.8936**   | 0.9182   | 0.8595   |

#### Why Optimize for Dice?
This model is the best general performer across metrics. Metrics like IoU tend to underrepresent performance on small structures due to their strict penalty on partial overlaps. Dice Coefficient was prioritized as the primary performance metric because it:
- Responds better to small object overlap
- Places less weight on boundary errors for large lesions
- Reflects clinical priorities: it’s better to detect a lesion imperfectly than to miss it entirely

#### Why Balance Recall - Precision?
This model prioritizes balancing false positives (over-detection) and false negatives (under-detection), achieving the highest F1 Score (harmonic mean of Precision and Recall). Suited for scenarios where over-detection and under-detection both carry risks. 

#### Why Optimize for Recall?
This model prioritizes minimizing false negatives to maximize sensitivity. It is designed to catch every possible lesion, erring on the side of over-detection. Suited for scenarios where missing lesion is unacceptable, even if it produces false positives.

---

## XAI Suite
- **Segmentation Mask Overlays**	    Direct visual comparison between predicted and ground-truth lesion boundaries. Helps assess accuracy intuitively.
- **Grad-CAM (Layer Visualization)**	Highlights which image regions most influenced the model’s decision. Useful for verifying focus on clinically relevant features. The full decision-making process can be visualized, end to end, from input to output.
- **Integrated Gradients**	          Attributive explanation that identifies input features most influential to output, with gradient integration for smoother interpretation.
- **Saliency Maps**	                  Highlights pixels with the highest gradient impact. Fast, intuitive way to visualize influence of specific regions.
- **Confusion Matrix**	              Quantifies true positives, false positives, false negatives, and true negatives. Offers insight into error distribution.
- **Superpixel Confidence Map**	      Visualizes model certainty across localized image regions. Supports risk estimation and diagnostic confidence.

---

## Image Processing Pipeline
A pre-processing and augmentation pipeline was developed to support both training-time and test-time experimentation. This enabled flexible trials of various techniques to enhance robustness, generalization, and fairness. While the final reported metrics were obtained on the **unaltered test set** (no pre-processing), several preprocessing techniques showed potential for improved generalization under specific conditions.

### Image-Mask Alignment
One of the key technical challenges in medical segmentation pipelines is ensuring perfect alignment between input images and their corresponding ground truth masks during data augmentation. Any mismatch — even a pixel shift — would corrupt the learning signal.

All geometric transformations (e.g., flipping, rotation, zoom) were applied using a shared random seed to ensure deterministic, mirrored changes to both images and masks. A post-transformation visual verification step was used during development to confirm correct alignment before training. Scripts were created to validate alignment before long training runs, which proved crucial for early-stage debugging.

### Modular Pipeline Design
The pipeline was built using a modular architecture, allowing toggling of transformations on or off, and setting boundaries (min/max intensity, rotation angles, etc.) from parameter inputs. This design enabled rapid iteration and testing.

### Techniques
- Brightness
- Color Saturation
- Contrast
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Dilation
- Edge Detection
- Erosion
- Gamma
- Gaussian Blur
- Gaussian Noise
- Horizontal Flip
- Hue Shift
- Rotation
- Vertical Flip
- Zoom

The final pipeline design prioritized reproducibility, modularity, and realism to enable efficient experimentation without compromising clinical plausibility.

---

## Key Development Milestones
These milestones reflect not only technical development but iterative experimentation, systematic validation, and a focus on real-world explainability.
### Problem Scoping & Constraints
- Defined clinical context, performance needs, and explainability requirements.
- Chose to exclude pretrained models and external datasets to enforce full transparency and auditability.
- Prioritized Dice Coefficient as the most clinically relevant metric.
### Data Review & Integrity
- Selected ISIC 2018 Challenge: Task 1 dataset for its clinical relevance and benchmark status.
- Maintained original training, validation, and test splits for reproducibility and fair comparisons.
### Model Architecture Design
- Built a custom U-Net with attention mechanisms from scratch.
- Incorporated layer normalization, dropout, and other regularization strategies to reduce overfitting.
### Modular Augmentation Pipeline
- Developed a modular image processing pipeline supporting toggled transformations.
- Solved and validated image-mask alignment through shared random seeds and post-transform verification.
- Enabled experimentation with transformations.
### Custom Loss Functions
- Implemented and compared Dice Loss, Tversky Loss, and a hybrid Dice–Tversky Loss to optimize for various clinical priorities.
- Aligned loss strategies with model variants.
### Initial Training
- Conducted initial training runs with aggressive logging, callbacks, and visualization.
- Verified base model performance using validation metrics prior to variant model fine-tuning.
### Variant Fine-Tuning
- Refined training parameters and loss functions for three model variants: Dice-Optimized, Balanced (F1), Recall-Optimized
- Tracked validation metrics to detect overfitting and guide early stopping.
### Evaluation & Benchmarking
- Computed all relevant performance metrics (Dice, IoU, Precision, Recall, Pixel Accuracy, F1 Score).
- Compared results across all three model variants.
- Used unaltered test set, no image preprocessing, for final benchmark reporting.
### Visualization & Verification
- Overlaid predicted segmentation masks on source images for visual inspection.
- Used this to detect potential post-processing errors.
### Preprocessing Experimentation
- Tested multiple preprocessing configurations and evaluated their impact on final test performance.
- Chose to report metrics without preprocessing to preserve direct comparability with other ISIC benchmark solutions.
### Explainability (XAI) Tools
- Implemented Grad-CAM, superpixel confidence mapping, saliency mapping, integrated gradients, and layer visualizations.
- Added interactive overlays and side-by-side views to improve interpretability.
- Tuned XAI tools to operate on either tensor or NumPy representations depending on compatibility.
### End to End Layer Visualization
- Enabled layer-level inspection to understand what features are extracted during encoding and decoding.
- Used this to build intuition, identify areas of model uncertainty, and visualize attention mechanisms at work.


---

## Future Work
### Performance Improvement
GlassBox XAI has several avenues for improving model performance with tradeoffs in interpretability, development time, on-device feasibility, and computational cost. Improving Dice performance from 0.875 to 0.9 is achievable if we expand training data beyond the relatively small ISIC 2018 set and optimize data augmentation during training. Improvement beyond this is achievable as well, but would require more time and computational complexity. For example, an ensemble combining a high-performing Attention U-Net with architecturally diverse models could match the performance of 2024 state-of-the-art solutions while remaining relatively lightweight compared to transformer-based systems.

- **Optimize Data Augmentation**
More advanced or domain-specific augmentation strategies could improve generalization. This would require more experimentation or domain expert collaboration.

- **Expand Training Data**
Incorporating more expert-annotated images from trusted sources (e.g., HAM10000) could improve model performance and reduce bias. This would be the most efficient way to improve metrics, and would have been prioritized in this iteration, but we constrained ourselves to using only the unaltered ISIC 2018 dataset for training.

- **Ensemble Models**
Combining the strengths of multiple models via ensembling or model averaging could improve overall performance, but increases inference time and complexity. While ensembling the three variant models in GlassBox XAI is technically possible, their shared architecture limits the diversity of learned representations, likely resulting in minimal gains. Greater benefits would be expected from ensembling architecturally diverse models.

- **Vision Transformers**
Transformer-based architectures have shown state-of-the-art performance in medical imaging tasks. Exploring these could improve segmentation performance. However, they generally require large-scale training data to outperform CNN-based models. Limited to relatively small datasets like ISIC 2018, their advantage may be reduced.

- **Pre-Trained Models**
Using pretrained encoders or models may accelerate convergence and improve performance, though it may reduce transparency and regulatory compliance if training data is not fully auditable. Additionally, most pretrained models require licensing agreements for full production deployment.

### Subject Matter Expert Collaboration
For GlassBox XAI to transition from proof-of-concept to clinical utility, collaboration with dermatology experts is essential.

- **Clinical Workflow Integration**
Partnering with dermatologists and clinical advisors can help validate real-world use cases, assess model performance in clinical settings, and identify opportunities for workflow integration and decision support.

- **UI/UX Development**
Building user interfaces tailored to clinical or educational use could improve usability for XAI visualizations. Integrating GlassBox XAI with clinical tools and EHR systems could further enhance its utility in real-world workflows.

- **Expanded XAI & Human-in-the-Loop (HITL) Tools**
Designing tools that allow experts to explore model reasoning, suggest corrections, and highlight edge cases could improve trust, enable targeted retraining, and accelerate model refinement through human-in-the-loop feedback.

---

## Tech Stack & Dependencies

This project was developed and executed entirely in a Jupyter Notebook environment using the following tools and libraries:

- Python - Core programming language
- Anaconda - Environment and dependency management
- Jupyter Notebook - Development and experimentation environment
- TensorFlow / Keras - Architecture, training, evaluation, metrics, pre-processing, Grad-CAM
- NumPy - Numerical operations, arrays, data handling
- Matplotlib / Seaborn - Data visualization
- Scikit-learn - Confusion matrices, metrics
- Scikit-image - Superpixel confidence mapping
- Albyumentations / OpenCV / SciPy - Image pre-processing, data augmentation, transformations
- Tqdm - Custom progress tracking
- ISIC 2018 Challenge Dataset - Benchmark dataset for training and evaluation

## Credits & References
This project uses data from the ISIC 2018: Task 1 – Lesion Segmentation challenge. All images and masks are publicly available, de-identified, and used here under the ISIC data use policy for research and educational purposes.

[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).

Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," 2018
https://arxiv.org/abs/1804.03999

## Author
**Jeffrey Robert Lynch** [LinkedIn](https://www.linkedin.com/in/jeffrey-lynch-350930348)

## License

This project is for educational and demonstration purposes only. For commercial use, please contact the author.
