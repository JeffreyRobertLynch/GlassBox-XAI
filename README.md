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

### Segmentation Features

- **Binary Segmentation Output:** The raw model output. A binary mask showing which pixels the model identifies as lesion and which it identifies as non-lesion.
- **Segmentation Overlay:** Predicted mask, or boundary decision, dimmed and laid over the original image for segmentation.  
- **Image Preprocessing Pipeline:** Fifteen toggleable preprocesing techniques, such as brightness adjustment and/or dilation, can be applied to images prior to segmentation to improve model performance. These techniques have been omitted for this demo. All visuals and reported metrics reflect performance on the unaltered, baseline dataset.
---

### Comparative Model Evaluation Features

- **Multi-model Segmentation Overlays:** Side-by-side visual comparison of all three model variants using the same image input. Highlights how each model handles precision, recall, and boundary decisions differently.
- **Multi-model Test Metrics:** Full evaluation across Dice, IoU, Precision, Recall, Pixel Accuracy, and F1 Score across all models.
- **Confusion Matrices:** Pixel-level false positive, false negative, true positive, and true negative rates for the entire test set across all models. Shows decision patterns & priorities.

---

### XAI Features

- **Confidence Maps:** Heatmap showing model confidence in its decisions across different regions of an image.
- **Saliency Maps:** Highlights which pixels were most influential to the model's decision.  
- **Integrated Gradients:** Quantifies each pixel's contribution to the final output by comparing it to a baseline. 
- **Grad-CAM Visualizations**: End-to-end visualization of all convolutional layers as a progression from input to output. Divided into encoder, attention, decoder, and output stages to show how segmentation decisions evolve through the network.

---

### Model Metrics

| Model              | Dice     | IoU      | Precision | Recall   | Pixel Accuracy | F1 Score |
|-------------------|----------|----------|-----------|----------|----------------|----------|
| **Dice-Optimized**   | **0.8751** | **0.8000** | **0.9028**  | 0.8291   | **0.9272**      | 0.8644   |
| **Balance-Optimized**| 0.8734   | 0.7925   | 0.8787    | 0.8564   | 0.9267         | **0.8674** |
| **Recall-Optimized** | 0.8573   | 0.7669   | 0.8280    | **0.8936** | 0.9182         | 0.8595   |

---

### Solution Summary

GlassBox XAI achieves **Dice 0.8751** and **IoU 0.8000**, meeting or exceeding top entries from the ISIC 2018 leaderboard—**without relying on pretrained models, external data, or ensemble methods**. While some modern models (2024–2025) may report Dice scores above 0.90, these often depend on substantial computational resources, cloud infrastructure, or proprietary datasets.

By contrast, GlassBox was developed **entirely from scratch**, trained on limited data, and runs **securely on local hardware**, with **end-to-end auditability**. It is suitable for real-world clinical deployment where **privacy, traceability, and ownership** are critical design factors.

In short, the system delivers **high-tier performance under realistic clinical constraints**, while also offering **powerful visual diagnostics**, **exceptional model transparency**, and **flexible, local deployability**—capabilities not often found in more opaque, resource-intensive solutions.

These trade-offs—and potential development paths for GlassBox XAI—are discussed further in the **Future Work** section.


GlassBox XAI achieves **Dice 0.8751** and **IoU 0.8000**, meeting or exceeding top entries from the ISIC 2018 leaderboard, **without relying on pretrained models, external data, or ensemble methods**. While some modern models (2024–2025) may report Dice scores above 0.90, these typically require computationally expensive techniques, substantial cloud infrastructure, or proprietary data pipelines.

By contrast, GlassBox was developed **entirely from scratch**, trained on limited data, and runs **securely on local hardware** with **end-to-end auditability**. It is suitable for real-world clinical deployment where **privacy, traceability, and model ownership** are critical design factors.

In short, this system delivers **high-tier performance under realistic clinical constraints**, while also offering **powerful visual diagnostics**, **exceptional model transparency**, and **flexibile, local deployability** not often found in more opaque, compute-heavy solutions.

These trade offs, and potential development paths for GlassBox XAI, are explored more thoroughly in the "Future Work" section.

---

## Key Features
These screenshots show the functionality of GlassBox XAI. All features can be used with all three models, but these screenshots will focus primarily on Model 1 (Dice-Optimized), to avoid redundancy. Three batches from the test set, containing 8 images each, are used to visualize features:

- Batch A (Average Metrics): Performance on this batch closely mirrors average performance on the full test set. This can be considered typical performance.
- Batch B (High Metrics): Performance on this batch is higher than average performance on the full test set. 
- Batch C (Low Metrics): Performance on this batch is lower than average performance on the full test set. 

### Core Segmentation Outputs
This section demonstrates how raw model predictions are turned into clear, interpretable visuals that support clinical or operational decision-making. The primary output is a binary segmentation mask, which highlights areas of interest, such as lesions, on a per-pixel basis.

To make these predictions human-readable, we overlay the segmentation mask on the original image, creating a visual output that can be used directly by clinicians or analysts. This is the real output of the system.

Ground truth masks, created by medical experts, are included here only for evaluation purposes. These are used to calculate objective performance metrics by comparing the model's output with the expert annotation.

#### Basic Segmentation Output - Model 1 - Batch A
Here we have the base image to be segmented, the expert annotated mask, and the models' outputted mask for comparison. 

![Model 1 - Segmentation Output - Batch A](output/base_model_1_batch_a_1.png)

#### Segmentation Overlay - Model 1 - Batch A
We take the model's output and create a dimmed overlay, placed over the original image. We do the same for the expert annotated mask for comparison. We also calculate performance metrics across this batch, for reference. This visual uses Batch A, representing typical performance.

![Model 1 - Segmentation Overlay - Batch A](output/metric_overlay_model_1_batch_a_1.png)

![Model 1 - Segmentation Overlay - Batch A](output/overlay_model_1_batch_a_1.png)

#### Segmentation Overlay - Model 1 - Batch B
This visual uses Batch B, representing high performance. The model's segmentation decision closely mirrors the expert annotation.

![Model 1 - Segmentation Overlay - Batch B](output/metric_overlay_model_1_batch_b_1.png)

![Model 1 - Segmentation Overlay - Batch B](output/overlay_model_1_batch_b_1.png)

#### Segmentation Overlay - Model 1 - Batch C
This visual uses Batch C, representing low performance. The model's segmentation decision mirrors the expert annotation for most images, but it isn't capturing the entirety of all lesions in this batch.

![Model 1 - Segmentation Overlay - Batch C](output/metric_overlay_model_1_batch_c_1.png)

![Model 1 - Segmentation Overlay - Batch C](output/overlay_model_1_batch_c_1.png)

### Comparative Model Evaluation
Building trust in AI systems requires more than clean outputs; it requires objective metrics. This section compares the performance of three models trained on the same task, showing how different training strategies affect generalization, reliability, and bias.

We use standard metrics like Dice coefficient, pixel accuracy, and confusion matrices to evaluate performance on a held-out test set.

Visual side-by-side comparisons further highlight the practical impact of model choice in terms of performance.

#### Multi-model Segmentation Comparison - All Models - Batch A
Here we visualize the segmentation decision of all 3 variant models for comparison. There are notable differences in performance between the variant models, other batches show greater divergence. 

![Multi-Model - Segmentation Output](output/metric_multi_model_batch_a_1.png)

![Multi-Model - Segmentation Output](output/multi_model_batch_a_1.png)

#### Model Performance Evaluation on Test Set - All Models - Test Set
These are the evaluation results of all 3 models on the full test set. Metric definitions can be found in the "Models" section. Model metrics are generally high for all three, but if we focus specifically on the trade off between Precision and Recall we can see that the primary difference in performance is how each model makes errors. Models err on the side of false negatives, balance, or false positives when classifying pixels. Metric definitions can be found in the "Models" section. 

##### Model 1 - Dice-Optimized
Precision is higher, Recall is lower.

![Model 1 - Segmentation Output](output/eval_model_1_1.png)

##### Model 2 - Balance-Optimized
Precision and Recall are balanced.

![Model 1 - Segmentation Output](output/eval_model_2_1.png)

##### Model 3 - Recall-Optimized
Precision is lower, Recall is higher.

![Model 1 - Segmentation Output](output/eval_model_3_1.png)


#### Confusion Matrices - All Models - Test Set
These are the confusion matrix results of all 3 models on the full test set. This calculates the model's correct and incorrect decisions on a pixel level across all images on the full test set. This shows model performance on a more granular level.

With these matrices, we can appreciate the difference in performance between these models. By focusing on the errors, false negatives and false positives, we will see a progression of false positives increasing and false negatives decreasing consistently. This does not indicate an overall increase of performance, all models are generally accurate and high-performance. Rather, each is specialized for different deployment environments with different sensitivities. This was achieved by using different custom loss functions for each model. 

##### Model 1 - Dice-Optimized
~2.5% of pixels in the test set are false positives and ~4.8% are false negatives. 

![Model 1 - Confusion Matrix](output/cm_model_1_1.png)

##### Model 2 - Balance-Optimized 
~3.3% of pixels in the test set are false positives and ~4.0% are false negatives.

![Model 2 - Confusion Matrix](output/cm_model_2_1.png)

##### Model 3 - Recall-Optimized 
~5.2% of pixels in the test set are false positives and ~3.0% are false negatives. 

![Model 3 - Confusion Matrix](output/cm_model_3_1.png)

### Interpretability & XAI
Understanding why a model makes a prediction is essential for trust, regulation, and real-world deployment in medical imaging AI.

This section showcases a suite of XAI techniques designed to help experts audit, debug, and validate predictions. It includes both pixel-level confidence and higher-level interpretability tools like saliency maps, integrated gradients, and Grad-CAM.

These tools allow practitioners to see what the model "attended to" when making decisions. This helps identify false positives, edge-case risks, or systemic biases. Whether you're debugging or certifying a model for clinical use, interpretability is critical.

#### Pixel Confidence Mapping - Model 1 - Batch A
This visualization overlays the segmentation result with color-coded superpixels, where each region is shaded based on the model's average confidence in that area. Warmer colors (red/yellow) indicate high certainty, while cooler colors (blue) highlight low-confidence regions.

This helps stakeholders:

- Interpret model trust at a localized level.

- Identify unreliable regions that may need review or manual verification.

- Support real-world decision-making by surfacing edge cases, ambiguity, or noise sensitivity in medical images.

Superpixel-based confidence mapping is especially useful in clinical or high-risk applications, where understanding how much the model trusts each part of its prediction can be just as important as the prediction itself.

![Model 1 - Confidence Mapping Output](output/conf_map_model_1_batch_a_1.png)

#### Saliency Mapping - Raw Logits (Unconstrained) - Model 1 - Batch A

![Model 1 - Saliency Mapping Output](output/sal_map_model_1_batch_a_1_raw.png)

##### Saliency Mapping - Sigmoid-Scaled (Constrained) - Model 1 - Batch A

![Model 1 - Saliency Mapping Output](output/sal_map_model_1_batch_a_1.png)

#### Integrated Gradients - Model 1 - Batch A

![Model 1 - Integrated Gradients Output](output/int_grad_model_1_batch_a_1.png)

#### Grad-CAM: Encoder Layers - Model 1 - Batch A

![Model 1 - Grad-CAM Encoder Layer Output](output/layer_enc_model_1_batch_a_1.png)

#### Grad-CAM: Attention Layers - Model 1 - Batch A

![Model 1 - Grad-CAM Attention Layer Output](output/layer_att_model_1_batch_a_1.png)

#### Grad-CAM: Decoder Layers - Model 1 - Batch A

![Model 1 - Grad-CAM Decoder Layer Output](output/layer_dec_model_1_batch_a_1.png)

#### Grad-CAM: Final Layer with Output - Model 1 - Batch A

![Model 1 - Grad-CAM Output Layer Output](output/layer_out_model_1_batch_a_1.png)

#### Grad-CAM: End-to-End Layer Visualization - Model 1 - Batch A


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
