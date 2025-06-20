# GlassBox-XAI
Attention U-Net for Medical Image Segmentation with XAI Suite  

## Overview
Advanced binary segmentation model trained on the ISIC 2018 dataset. Built for high performance and full transparency via a robust explainable AI (XAI) suite. 

Three models were trained for different specializations. All three models can be easily used for image processing and their outputs compared directly to one another in terms of visuals and metrics.

## Key Features
- Custom **Attention U-Net** architecture for segmentation.
- End-to-end workflow: preprocessing, training, evaluation, and XAI.
- High production-level metrics (Dice, IoU, Recall, Precision, F1).
- Model variants to explore Dice optimization vs. Recall optimization vs. balanced performance.
- XAI tools: Grad-CAM, Integrated Gradients, Saliency Maps, and more.
- Focused on **regulatory compliance**, decision support, and **human-in-the-loop** workflows.

## Results Snapshot - Model 1 - Dice Optimized
| Metric         | Score    |
|----------------|----------|
| Dice Coefficient | 0.8751   |
| IoU            | 0.8000   |
| Precision      | 0.9028   |
| Recall         | 0.8291   |
| Accuracy       | 0.9272   |
| F1 Score       | 0.8644   |

- **Dice Coefficient**	            Measures overlap between prediction and ground truth. More tolerant of small errors, ideal for clinical segmentation.
- **Intersection over Union (IoU)**	Stricter overlap metric. Penalizes all misalignments equally, which may underrepresent small lesions.
- **Recall (Sensitivity)**	        Measures how many actual positives were correctly identified. Key to avoiding missed disease areas.
- **Precision**	                    Measures how many predicted positives were actually correct. Helps reduce false positives.
- **F1 Score**	                    Harmonic mean of precision and recall. Balances false negatives and false positives.
- **Pixel Accuracy**	              Overall proportion of correctly labeled pixels. Can be misleading in class-imbalanced datasets like medical imaging.

## Why Dice Was Prioritized
In medical image segmentation, missing small lesions can have serious clinical consequences. Metrics like IoU tend to underrepresent performance on small structures due to their strict penalty on partial overlaps.

Dice Coefficient was prioritized as the primary performance metric because it:

- Responds better to small object overlap
- Places less weight on boundary errors for large lesions
- Reflects clinical priorities: it’s better to detect a lesion imperfectly than to miss it entirely

**All metrics were reported for benchmarking, but Dice is emphasized due to its greater alignment with decision-support needs and regulatory expectations.**

## Tech Stack
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

## XAI Suite
- **Segmentation Mask Overlays**	    Direct visual comparison between predicted and ground-truth lesion boundaries. Helps assess accuracy intuitively.
- **Grad-CAM (Layer Visualization)**	Highlights which image regions most influenced the model’s decision. Useful for verifying focus on clinically relevant features. The full decision-making process can be visualized, end to end, from input to output.
- **Integrated Gradients**	          Attributive explanation that identifies input features most influential to output, with gradient integration for smoother interpretation.
- **Saliency Maps**	                  Highlights pixels with the highest gradient impact. Fast, intuitive way to visualize influence of specific regions.
- **Confusion Matrix**	              Quantifies true positives, false positives, false negatives, and true negatives. Offers insight into error distribution.
- **Superpixel Confidence Map**	      Visualizes model certainty across localized image regions. Supports risk estimation and diagnostic confidence.

## Coming Soon
A recorded **tech demo** will walk through:
- Code structure
- Key engineering decisions
- How to build explainable, regulatory-ready medical AI

## Author
[LinkedIn](https://www.linkedin.com/in/jeffrey-lynch-350930348)

## License

This project is for educational and demonstration purposes only. For commercial use, please contact the author.
