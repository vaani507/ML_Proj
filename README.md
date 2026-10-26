# ML_Proj

Problem:
Solar photovoltaic cells can develop defects such as micro-cracks, broken grids, and other structural abnormalities during manufacturing and handling. These defects can negatively affect the efficiency, reliability, and lifespan of solar panels. Traditional quality-control systems often depend on manual visual inspection, which can be time-consuming, subjective, and unreliable when defects are small or difficult to distinguish from normal cell structures.
The challenge is therefore to automatically identify defective solar cells and accurately determine the location and extent of defects from electroluminescence (EL) images before the cells are integrated into solar panels.
Proposed ML Solution:
 We will develop a deep-learning-based solar cell inspection system using EL images of photovoltaic cells. The system will combine image classification and semantic segmentation to identify defective cells and localize their damaged regions.
The project will have two main ML tasks:
1. Defect Classification:
 Use transfer-learning-based CNN models such as ResNet18, ResNet50, or EfficientNet to classify solar-cell images according to whether they contain defects and, where applicable, identify the type of defect. Data augmentation and class-imbalance handling will be explored to improve the model's ability to recognize less frequent defect patterns.
2. Defect Segmentation:
 Use U-Net or a ResNet-encoder U-Net architecture to generate pixel-level masks of defective regions. The segmentation model will identify the exact location and shape of cracks, broken grids, or other defects rather than simply assigning a defective/non-defective label.
The models will be evaluated using accuracy, precision, recall, F1-score, Dice coefficient, and Intersection over Union (IoU). Special emphasis will be placed on recall and segmentation IoU because missing a defect or incorrectly estimating its affected area can lead to defective cells being accepted during manufacturing.
Why this is an interesting ML project:
 Instead of simply classifying a solar cell as defective or healthy, the system learns visual patterns associated with manufacturing defects and determines where the defect occurs and how much of the cell is affected.
The project combines transfer learning, image classification, semantic segmentation, data augmentation, class-imbalance handling, and explainable computer vision into a single automated quality-control system. The comparison between a conventional CNN classifier and a U-Net-based segmentation model also provides an opportunity to investigate whether pixel-level localization provides more useful information than classification alone.
