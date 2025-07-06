# GlassBox-XAI
Attention U-Net for Medical Image Segmentation with XAI Suite  

**Disclaimer: This is not a medical device and not intended for diagnosis or treatment. The results shown reflect performance on the curated ISIC 2018 test set (~1,000 images) and have not been evaluated for clinical use.**

---

## Problem Domain
This project focuses on **automated binary segmentation of skin lesions** in medical images, a fundamental step toward early melanoma detection. Accurate boundary delineation (lesion vs. background) is essential for:

- **Risk Triage:** Flagging high-risk areas for deeper evaluation.

- **Quantitative Tracking:** Monitoring lesion growth or regression over time.

- **Clinical Workflows:** Integrating into Computer-Aided Diagnosis (CAD) tools.

Unlike simple classification, which only states "lesion present," accurate segmentation reveals shape irregularities and precise border limits, aiding biopsy guidance and treatment decisions.

### Why Explainability (XAI) Is Critical in Healthcare AI
Building transparent AI is essential, not optional, for clinical adoption:

- **Clinician Trust & Collaboration:** Tools like saliency and Grad‑CAM heatmaps help clinicians understand how and why the model made a specific decision, including the model's confidence level at the pixel level. This supports human-in-the-loop (HITL) workflows and informed decision-making. 

- **Regulatory & Ethical Compliance:** XAI is increasingly required for medical device approval and supports bias detection, auditability,  and safe deployment.

- **Model Validation & Iterative Improvement:** Visual explanations enable SME feedback loops, critical for clinical trials, continuous improvement, and device certification.

- **Real-world Utility:** Studies show XAI-enabled tools increase trust and performance in clinical settings.

---

## Solution Overview

### Objective & Scope
- Build a from-scratch solution to ISIC 2018, Task 1: Binary Segmentation, that meets or exceeds the performance of top 2018 leaderboard solutions.
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
- **Model Variants:** Dice-Optimized prioritizes boundary accuracy, Recall-Optimized prioritizes minimizing false negatives, and Balanced prioritizes minimizing false negatives and false positives overall.
- **Evaluation:** All three models were evaluated and compared using the following metrics: Dice Coefficient, Intersection of Union, Precision, Recall, Pixel Accuracy, and F1 Score. Metric definitions can be found in the "Metric Definition" section. 
- **XAI Integration:** After initial performance evaluation, XAI features were developed, tested, and integrated to evaluate model decision-making at a more granular level. These features are visualized and explained in the "Key Features" and "XAI Suite" sections.

### Results
- **Performance:** Dice-Optimized (Model 1) achieved a Dice Score of ~0.875 and an IoU score of ~0.8. The other 2 models have similarly high performance metrics within their specializations. This meets or exceeds the performance of 2018 solutions to the problem. Full information on performance for all models can be seen in the "Key Features" and "Models" sections.
- **Auditability & Reproducibility:** Transparent, portable, and generalizable workflow for medical image segmentation with XAI. Clear audit trail for training data, test data, and evaluation.
- **Security:** Entire process remains on-device within a controlled environment to protect PHI from data leakage or breaches.
- **XAI Integration:** Each model's base output is a mask, or outline, delineating what part of the image contains a lesion and what part of the image does not contain a lesion. This can be used for detecting lesions, measuring the growth or retraction of lesions over time, or providing focus for experts. Beyond this, XAI features provide multi-model comparisons, confidence matrices, confidence maps, saliency maps, Grad-CAM, integrated gradients, and full end to end layer visualization. These visualization features, described in full in the "XAI Suite" section, make model decisions transparent, trustworthy, and reviewable. GlassBox AI is not a "black box" model.

### Conclusions
GlassBox AI proves that production-level performance segmentation models can be trained, evaluated, and deployed with modest computational resources. This supports local development - models can be trained on local population data to mitigate bias while maintaining information security. This also supports wide deployment - any facility with modest computational resources can locally deploy this solution without relying on complex, and potentially insecure, data chains. Likewise, model ownership and auditability are preserved by avoiding pretrained models. Further, robust XAI features make this solution transparent, trustworthy, and aligned with clinical trials and regulatory evaluation.

Though GlassBox AI compares favorably with top 2018 solutions in performance and transparency, it is a 2024 solution built on years of progress. 2018 solutions did not have access to an additional 5 years of technical innovation and hindsight. In 2024, some newer solutions outperform GlassBoxAI's metrics - often at the cost of data security, computational complexity, transparency, auditability, and other factors. These new methodologies and techniques, and their trade-offs, will be explored in the "Future Work" section. 

---

## Key Features

These screenshots show the functionality of GlassBox AI. All features can be used with all three models, but these screenshots will focus primarily on Model 1 (Dice-Optimized), to avoid redundancy.

### Batch Segmentation & Visualization - Model 1

### Multi-model Comparison with Dimmed Overlay - Model 1, Model 2, Model 3

### Model Performance Evaluation on Test Set - Model 1, Model 2, Model 3

### Confusion Matrices - Model 1

### Pixel Confidence Heatmap - Model 1

### Saliency Map - Model 1

### Integrated Gradients - Model 1

### Grad-CAM: Encoder Layers - Model 1

### Grad-CAM: Attention Layers - Model 1

### Grad-CAM: Decoder Layers - Model 1

### Grad-CAM: Final Layer with Output - Model 1

### Grad-CAM: End-to-End Layer Visualization - Model 1

---

## Data
Curation.
Anonymized. 
No PHI. 
2600 images & masks.
Training.
Testing.

---

## Models & Metrics

### Architecture
- All models were trained from scratch on freshly initialized custom architecture.
- All models use the same U-Net architecture with added attention mechanisms.
- Layer normalization, regularization, dropout were implemented. 
- Each model was fine-tuned using custom loss functions: Dice Loss, Tversky Loss, and Hybrid Dice-Tversky Loss.

---

### Performance Metrics
- **Dice Coefficient:**	            Measures overlap between prediction and ground truth. More tolerant of small errors in large lesions, less tolerant of errors in small lesions. Ideal for clinical segmentation.
- **Intersection over Union (IoU):**	Stricter overlap metric. Penalizes all misalignments equally, which may underrepresent small lesions.
- **Recall (Sensitivity):**	        Measures how many true positives were correctly identified. Key to avoiding missed disease areas.
- **Precision:**	                    Measures how many true negatives were correctly identified. Helps reduce false positives.
- **F1 Score:**	                    Harmonic mean of precision and recall. Balances false negatives and false positives.
- **Pixel Accuracy:**	              Overall proportion of correctly labeled pixels. Can be misleading in class-imbalanced datasets like medical imaging.

### Model Performance Comparison

| Model        	 | Dice     | IoU      | Precision| Recall   |Pixel Acc | F1 Score |
|----------------|----------|----------|----------|----------|----------|----------|
| Dice-Optimized | 0.8751   | 0.8000   | 0.9028   | 0.8291   | 0.9272   | 0.8644   |
| Balanced       | 0.8735   | 0.7926   | 0.8790   | 0.8561   | 0.9267   | 0.8674   |
|Recall-Optimized| 0.8575   | 0.7672   | 0.8286   | 0.8932   | 0.9184   | 0.8597   |

#### Why Optimize for Dice?
Missing small lesions can have serious clinical consequences. Metrics like IoU tend to underrepresent performance on small structures due to their strict penalty on partial overlaps. Dice Coefficient was prioritized as the primary performance metric because it:
- Responds better to small object overlap
- Places less weight on boundary errors for large lesions
- Reflects clinical priorities: it’s better to detect a lesion imperfectly than to miss it entirely

**All metrics were reported for benchmarking, but Dice is emphasized due to its greater alignment with decision-support needs and regulatory expectations.**

#### Why Balance Recall - Precision?
This model prioritizes balancing false positives and false negatives, making its overall F1 Score (average of Precision and Recall) slightly higher than the other two models.

#### Why Optimize for Recall?
This model prioritizes minimizing false negatives above all else. This sensitivity also makes it the most likely to produce false positives as a tradeoff.  

---

## XAI Suite
- **Segmentation Mask Overlays**	    Direct visual comparison between predicted and ground-truth lesion boundaries. Helps assess accuracy intuitively.
- **Grad-CAM (Layer Visualization)**	Highlights which image regions most influenced the model’s decision. Useful for verifying focus on clinically relevant features. The full decision-making process can be visualized, end to end, from input to output.
- **Integrated Gradients**	          Attributive explanation that identifies input features most influential to output, with gradient integration for smoother interpretation.
- **Saliency Maps**	                  Highlights pixels with the highest gradient impact. Fast, intuitive way to visualize influence of specific regions.
- **Confusion Matrix**	              Quantifies true positives, false positives, false negatives, and true negatives. Offers insight into error distribution.
- **Superpixel Confidence Map**	      Visualizes model certainty across localized image regions. Supports risk estimation and diagnostic confidence.

---

## Modular Image Processing
Challenge of aligning images and masks during flipping, rotation, and zoom transformations. Post augmentation, Pretraining testing.
Random Transformations within narrow boundaries for realistic image augmentations.
Zoom
Rotation
Flip
Saturation
Hue
Contrast
Gamma
Brightness
Noise
Blur
Dilation
Erosion
Edge Detection
Histogram Equalization
CLAHE

---

## Key Development Milestones
### Review Problem & Limitations
### Review Data
### Model Architecture Design
### Modular Data Augmentation
### Custom Loss Functions
### Initial Training
### Final Tuning
### Evaluation
### Mask Overlay Visualization
### Preprocessing Experimentation
### XAI Visualizations
### End to End Layer Visualization

---

## Future Work
### Performance Improvement
Optimize Data Augmentation
Ensemble Models
Vision Transformers
Expand Training Data
Pre-Trained Models

### Subject Matter Expert Review & Collaboration
Integration with Workflows
UI and UX Development
Additional XAI and HITL Features

---

## Tech Stack & Dependencies
- Python
- Anaconda
- Jupyter Notebook
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Seaborn
- Tqdm
- ISIC 2018 Challenge Dataset

## Credits & References

## Author
**Jeffrey Robert Lynch** [LinkedIn](https://www.linkedin.com/in/jeffrey-lynch-350930348)

## License

This project is for educational and demonstration purposes only. For commercial use, please contact the author.
