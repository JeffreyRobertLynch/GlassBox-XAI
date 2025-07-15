# GlassBox-XAI
Attention U-Net for Medical Image Segmentation with XAI Suite  

**Disclaimer: This project is for research and demonstration purposes only. It is not a medical device and is not intended for clinical use, diagnosis, or treatment. All medical images are from the anonymized, public ISIC 2018 dataset and contain no PHI.**

---

## Problem Domain
This project focuses on **automated binary segmentation of skin lesions** in medical images, a fundamental step toward early melanoma detection. Accurate boundary delineation (lesion vs. background) is essential for:

- **Risk Triage:** Flagging high-risk areas for deeper evaluation.

- **Quantitative Tracking:** Monitoring lesion growth or regression over time.

- **Clinical Workflows:** Integrating into Computer-Aided Diagnosis (CAD) tools.

Unlike simple classification, which only states "lesion present", accurate segmentation reveals shape irregularities and precise border limits, aiding biopsy guidance and treatment decisions.

### Why Explainability (XAI) Is Critical in Healthcare AI
Building transparent AI is essential, not optional, for clinical adoption:

- **Clinician Trust & Collaboration:** Tools like saliency and Grad‑CAM heatmaps help clinicians understand how and why the model made a specific decision, including the model's confidence level at the pixel level. This supports human-in-the-loop (HITL) workflows and informed decision-making. 

- **Regulatory & Ethical Compliance:** XAI is increasingly required for medical device approval and supports bias detection, auditability, and safe deployment.

- **Model Validation & Iterative Improvement:** Visual explanations enable SME feedback loops, critical for clinical trials, continuous improvement, and device certification.

- **Real-world Utility:** Studies show XAI-enabled tools increase trust and performance in clinical settings.

---

## Solution Overview

### Objective & Scope
- Build a from-scratch solution to ISIC 2018, Task 1: Binary Segmentation, that matches or outperforms the top 2018 leaderboard solutions.
- Adhere to the ISIC 2018 challenge dataset splits (Training, Validation, Test) for quantitative performance benchmarking and impose additional limitations for auditability and security.
- Fine-tune three model variants (Dice-Optimized, Recall-Optimized, Balanced) to support flexible deployment goals.
- Maintain full auditability, ownership, and replication using a custom architecture and no pretrained models.
- Develop, train, test, and deploy on local hardware to ensure optimal Private Health Information (PHI) security.  
- Integrate a robust XAI Suite to support clinical trust, regulatory compliance, human-in-the-loop workflows, and iterative improvement.

### Resources & Limitations
- **Hardware:** Modest home-lab setup (NVIDIA RTX 3080 GPU, 32 GB RAM, AMD 3700X CPU) to ensure closed system for PHI security and wide local deployment. 
- **Timeframe:** Solo execution in ~160 Hours, approximately one month. Includes research, development, and evaluation. Excludes time spent on full documentation.
- **Data:** Used only publicly available and anonymized ISIC 2018 dataset, no external or proprietary PHI.
- **Security:** Entire pipeline (training, evaluation, and deployment) performed locally to protect data integrity.
- **Reproducibility:** Portable and generalizable solution using custom-built pipeline, documented methodologically, transparent metrics, and consistent evaluation logic.
- **Constraints:** No ensembles, transfer/pretrained learning, cloud usage, external datasets, or GPU clusters. Feasible for realistic local development and deployment with modest computational resources.  

### Methodology
- **Model Architecture:** Custom-built U-Net architecture with added attention mechanisms, dropout, and regularization. 
- **Training Pipeline:** Modular data augmentation, augmentation testing, custom loss functions, callbacks, and fine-tuning at lower learning rate for variant specialization.  
- **Model Variants:**
  - *Dice-Optimized* — maximizes boundary accuracy   
  - *Balanced* — balances false positives and false negatives
  - *Recall-Optimized* — minimizes false negatives 
- **Evaluation:** Dice Score (Dice), Intersection of Union (IoU), Precision, Recall, Pixel Accuracy, and F1 Score. Metric definitions can be found in the "Models" section. 
- **XAI Integration:** XAI tools inspect internal decision-making at a more granular level for interpretability. These features are visualized and explained in the "Key Features" and "XAI Suite" sections.

### Results
- **Performance:** *Dice-Optimized* model achieved a Dice Score of ~0.875 and an IoU score of ~0.8, matching or outperforming top 2018 solutions. Other models achieved similarly high performance within their optimization goals. Full information on performance for all models can be seen in the "Key Features" and "Models" sections.
- **Auditability & Reproducibility:** Transparent, portable, and transferrable workflow with clear audit trail.
- **Security:** Entire process remains on-device, protecting PHI from data leakage or breaches.
- **XAI Integration:** GlassBox XAI is not a "black box" model. Each model’s base output is a binary mask outlining lesion vs. non-lesion regions. This can be used for detecting lesions, measuring the growth or retraction of lesions over time, and supporting clinical & surgical decisions. Beyond this, XAI features provide multi-model comparisons, confusion matrices, confidence maps, saliency maps, Grad-CAM, integrated gradients, and full end to end layer visualization. These visualization features, described in full in the "XAI Suite" section, enhance interpretability, trust, and regulatory compliance.

### Conclusions
GlassBox XAI demonstrates that **production-level segmentation performance** can be achieved on modest hardware without the use of pretrained models, black box limitations, or privacy trade-offs. This makes the solution deployable in local settings (e.g., clinics, research hospitals) where data security and transparency are essential.

GlassBox XAI delivers performance on par with or exceeding top 2018 leaderboard entries, while improving transparency, auditability, and deployment feasibility. However, it was produced in 2024 and some 2024-era systems may surpass GlassBox XAI's metrics using larger models, expanded training data, or ensemble learning — but often at the cost of transparency, auditability, security, and accessibility.

The trade-offs of these techniques are discussed in the "Future Work" section.

---

## Data Ethics & Full Disclaimer
**This project is for research and demonstration purposes only. It is not a medical device and is not intended for clinical use, diagnosis, or treatment.**

**All medical images shown below are part of the ISIC 2018 Challenge Dataset, a publicly released and anonymized dataset curated for research. No private health information (PHI) is present.**

**Results presented are generated automatically by deep learning models, and their accuracy is benchmarked using the ISIC-provided test set, not through real-world clinical evaluation.**

**This solution has not undergone clinical validation or regulatory review.**

**Visualizations (e.g., masks, heatmaps, Grad-CAM) are interpretability tools, not clinical indicators.**

**This project does not offer diagnostic guidance and should not be used as such.**

Please refer to the ISIC 2018 dataset license and usage guidelines for full dataset attribution and terms of use.

---

## Key Features
These screenshots show the functionality of GlassBox XAI. All features can be used with all three models, but these screenshots will focus primarily on Model 1 (Dice-Optimized), to avoid redundancy. Three batches fro mthe test set, containing 8 images each, are used to visualize features:

- Batch A (Average Metrics): Performance on this batch closely mirrors average performance on the full test set. This can be considered typical performance.
- Batch B (High Metrics): Performance on this batch is higher than average performance on the full test set. 
- Batch C (Low Metrics): Performance on this batch is lower than average performance on the full test set. 

### Core Segmentation Outputs
This section demonstrates how raw model predictions are turned into clear, interpretable visuals that support clinical or operational decision-making. The primary output is a binary segmentation mask, which highlights areas of interest, such as lesions, on a per-pixel basis.

To make these predictions human-readable, we overlay the segmentation mask on the original image, creating a visual output that can be used directly by clinicians or analysts. This is the real output of the system.

Ground truth masks, created by medical experts, are included here only for evaluation purposes. These are used to calculate objective performance metrics by comparing the model's output with the expert annotation.

#### Basic Segmentation Output - Model 1 - Batch A
![Model 1 - Segmentation Output](output/base_output/base_model_1_batch_78_1.png)

#### Segmentation Overlay - Model 1 - Batch A

#### Segmentation Overlay - Model 1 - Batch B

#### Segmentation Overlay - Model 1 - Batch C

### Comparative Model Evaluation

#### Multi-model Segmentation Comparison - All Models - Batch A

#### Model Performance Evaluation on Test Set - All Models - Test Set

#### Confusion Matrices - All Models - Test Set

### Interpretability & XAI

#### Pixel Confidence Mapping - Model 1 - Batch A
This visualization overlays the segmentation result with color-coded superpixels, where each region is shaded based on the model's average confidence in that area. Warmer colors (red/yellow) indicate high certainty, while cooler colors (blue) highlight low-confidence regions.

This helps stakeholders:

- Interpret model trust at a localized level.

- Identify unreliable regions that may need review or manual verification.

- Support real-world decision-making by surfacing edge cases, ambiguity, or noise sensitivity in medical images.

Superpixel-based confidence mapping is especially useful in clinical or high-risk applications, where understanding how much the model trusts each part of its prediction can be just as important as the prediction itself.

#### Saliency Mapping - Model 1 - Batch A

#### Integrated Gradients - Model 1 - Batch A

#### Grad-CAM: Encoder Layers - Model 1 - Batch A

#### Grad-CAM: Attention Layers - Model 1 - Batch A

#### Grad-CAM: Decoder Layers - Model 1 - Batch A

#### Grad-CAM: Final Layer with Output - Model 1 - Batch A

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

### Metrics Explained
- **Dice Coefficient** measures how well the predicted lesion area overlaps with the actual lesion. IT is especially sensitive to errors in small lesions and is a standard metric in medical image segmentation.
- **Intersection over Union (IoU)**	is a stricter version of Dice. Penalizes all boundary mismatches equally, which can underrepresent small or subtle lesions.
- **Recall (Sensitivity)** measures how well the model captures all lesion areas. Crucial in medical contexts, where missing even a small lesion could be risky.
- **Precision**	measures how many predicted lesion areas were correct. Helps avoid false alarms that could lead to unnecessary concern or procedures.
- **F1 Score** balances precision and recall. Useful when both missing and over-detecting lesions have consequences.
- **Pixel Accuracy** shows the overall percentage of correctly labeled pixels. Can be misleading in medical imaging, where lesions may be a small part of the image.

**All metrics contribute to measuring performance, but Dice is emphasized due to its greater alignment with decision-support needs and regulatory expectations.**

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
