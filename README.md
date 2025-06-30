# GlassBox-XAI
Attention U-Net for Medical Image Segmentation with XAI Suite  

---

## Overview
Advanced binary segmentation model trained on the ISIC 2018 dataset. Trained from scratch with a custom Attention U-Net architecture for high performance and full transparency via a robust explainable AI (XAI) suite. Three models were trained for different specializations. All three models can be easily used for image processing and their outputs compared directly to one another in visuals and metrics.

**Disclaimer: This is not a medical device and not intended for diagnosis or treatment. The results shown reflect performance on the curated ISIC 2018 test set (~1,000 images) and have not been evaluated for clinical use.**
---

## Problem Domain
**International Skin Imaging Collaboration (ISIC) 2018 – Lesion Boundary Segmentation:**
This project addresses the automated segmentation of skin lesion boundaries in dermoscopic imagery, a critical step toward identifying melanoma. Automated segmentation can increase access to diagnostic support and standardize analysis across clinicians. Beyond high-performance segmentation accuracy, this solution emphasizes **full transparency** through an XAI Suite featuring confidence maps, end-to-end layer visualization, multi-model comparison, and other techniques. Experts can follow, compare, and evaluate the models' decision-making process, supporting: 
- **Clinician trust**
- **Human-in-the-loop (HITL) workflows**
- **Expert review & iterative improvement**
- **Clinical trials & regulatory compliance**
- **Full XAI details can be found in the "XAI Suite" section**

### Clinical & Diagnostic Applications
- **Risk Triage:** Flagging suspicious areas for further analysis.
- **Quantitative Tracking:** Precisely monitor lesion growth or regression.
- **Supporting Diagnosis Workflows:** Serves as a key component in Computer-Aided Diagnosis (CAD) systems.

### Model Optimization - Three Variants 
- **Dice-Optimized:** Maximizes segmentation overlap with test set.
- **Recall-Optimized:** Focused on sensitivity to prioritize reducing false negatives.
- **Recall-Precision Balanced:** Midpoint between recall (false negatives) and precision (false positives) for conservative clinical use.

### ISIC 2018 Challenge Dataset:
The ISIC 2018 dataset (~2,600 images) is a high-quality, expert-curated public benchmark for lesion segmentation. Test set performance here is a widely accepted proxy for production-level performance and clinical trial suitability. Full dataset details are in the **“Data”** and **“Credits”** sections. This solution's performance aligns closely with top-performing models in open-source benchmarks. Full information on performance metrics for each model, and the relevance of each metric, can be found in the **"Models"** and **"Performance Metrics"** sections.

---
## Key Features
- Custom **Attention U-Net** architecture for segmentation.
- End-to-end workflow: preprocessing, training, evaluation, and XAI.
- High production-level metrics (Dice, IoU, Recall, Precision, F1).
- Model variants to explore Dice optimization vs. Recall optimization vs. balanced performance.
- XAI tools: Grad-CAM, Integrated Gradients, Saliency Maps, and more.
- Focused on **regulatory compliance**, decision support, and **human-in-the-loop** workflows.

---

## Data

---

## Models & Metrics
All models were trained from scratch, on custom architecture..
No pretrained models...
Different Loss functions...

---

### Performance Metrics
- **Dice Coefficient**	            Measures overlap between prediction and ground truth. More tolerant of small errors, ideal for clinical segmentation.
- **Intersection over Union (IoU)**	Stricter overlap metric. Penalizes all misalignments equally, which may underrepresent small lesions.
- **Recall (Sensitivity)**	        Measures how many actual positives were correctly identified. Key to avoiding missed disease areas.
- **Precision**	                    Measures how many predicted positives were actually correct. Helps reduce false positives.
- **F1 Score**	                    Harmonic mean of precision and recall. Balances false negatives and false positives.
- **Pixel Accuracy**	              Overall proportion of correctly labeled pixels. Can be misleading in class-imbalanced datasets like medical imaging.

### Model 1 - Dice Optimized
| Metric         | Score    |
|----------------|----------|
| Dice Coefficient | 0.8751   |
| IoU            | 0.8000   |
| Precision      | 0.9028   |
| Recall         | 0.8291   |
| Accuracy       | 0.9272   |
| F1 Score       | 0.8644   |

#### Why Optimize for Dice?
In medical image segmentation, missing small lesions can have serious clinical consequences. Metrics like IoU tend to underrepresent performance on small structures due to their strict penalty on partial overlaps. Dice Coefficient was prioritized as the primary performance metric because it:
- Responds better to small object overlap
- Places less weight on boundary errors for large lesions
- Reflects clinical priorities: it’s better to detect a lesion imperfectly than to miss it entirely

### Model 2
| Metric         | Score    |
|----------------|----------|
| Dice Coefficient | 0.8751   |
| IoU            | 0.8000   |
| Precision      | 0.9028   |
| Recall         | 0.8291   |
| Accuracy       | 0.9272   |
| F1 Score       | 0.8644   |


#### Why Optimize for Recall?

### Model 3
| Metric         | Score    |
|----------------|----------|
| Dice Coefficient | 0.8751   |
| IoU            | 0.8000   |
| Precision      | 0.9028   |
| Recall         | 0.8291   |
| Accuracy       | 0.9272   |
| F1 Score       | 0.8644   |

#### Why Balance Recall - Precision?

In medical image segmentation, missing small lesions can have serious clinical consequences. Metrics like IoU tend to underrepresent performance on small structures due to their strict penalty on partial overlaps. Dice Coefficient was prioritized as the primary performance metric because it:

- Responds better to small object overlap
- Places less weight on boundary errors for large lesions
- Reflects clinical priorities: it’s better to detect a lesion imperfectly than to miss it entirely

**All metrics were reported for benchmarking, but Dice is emphasized due to its greater alignment with decision-support needs and regulatory expectations.**
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

---

## Key Development Milestones

---

## Future Work

---

## Tech Stack & Dependencies
- Python
- Anaconda
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Seaborn
- ISIC 2018 Challenge Dataset

## Credits & References

## Author
**Jeffrey Robert Lynch** [LinkedIn](https://www.linkedin.com/in/jeffrey-lynch-350930348)

## License

This project is for educational and demonstration purposes only. For commercial use, please contact the author.
