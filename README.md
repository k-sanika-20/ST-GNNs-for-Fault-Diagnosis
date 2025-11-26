## Spatio-Temporal Graph Neural Networks for Multi-Sensor Fusion in Fault Diagnosis

This project notebook contains the implementation and analysis for the Bachelor of Technology thesis investigating the use of **Spatio-Temporal Graph Neural Networks (ST-GNNs)** for multi-sensor data fusion in rotating machinery **fault diagnosis**.

The entire model training and evaluation process was conducted using **Kaggle Notebooks** with GPU acceleration.

---

### Project Goal and Core Hypothesis

The central problem addressed is that traditional multi-sensor data fusion methods often fail to model the inherent **physical and spatial relationships** between sensors, leading to a loss of important diagnostic information.

The core hypothesis tested was: Can an ST-GNN, operating on a simple **physically-grounded graph** (where sensors are nodes and their connection is an edge), provide a more robust and generalizable fault diagnosis compared to standard methods?

#### Graph Formulation

The multi-sensor system was modeled as a graph-level classification task:
* **Nodes:** Represent the physical sensors (Drive End - DE, and Fan End - FE accelerometers).
* **Edges:** A single undirected edge represents the physical connection between the sensors via the motor housing.
* **Architecture:** A hybrid ST-GNN combining a **Recurrent Neural Network (RNN)** (specifically LSTM) for temporal modeling and a **Graph Neural Network (GNN)** (specifically GAT) for spatial dependency capturing.

---

### Methodology and Comparative Models

All models were evaluated using the **Case Western Reserve University (CWRU) bearing dataset**. A strict **mixed-load training strategy** was used: training on 0, 1, and 2 HP load data, and testing exclusively on a completely **unseen 3 HP load data** to measure generalization and robustness.

| Model | Input Type / Feature Representation | Graph Edge Weight | Unseen-HP Accuracy (Robustness) |
| :--- | :--- | :--- | :--- |
| **1D-CNN (Benchmark)** | Raw Signal (1024 points) | N/A (Standard CNN fusion) | **97.82%** |
| **ST-GNN v1** | Raw Signal (1024 points) | Unweighted (Binary) | **78.99%** |
| **ST-GNN v2** | **11 Handcrafted Features** (Time/Frequency) | Unweighted (Binary) | **92.56%** |
| **ST-GNN v3** | 11 Handcrafted Features | **Correlation-Weighted** (Absolute Pearson's *r*) | **94.90%** |

---

### Key Findings

1.  **GNNs Require Feature Engineering:** ST-GNN v1 (Raw Signals) struggled (78.99%). Switching to **ST-GNN v2 (Handcrafted Features)** resulted in a major performance leap to **92.56%**, validating the necessity of informative feature representation.
2.  **Weighted Edges Improve Fusion:** Explicitly incorporating the physical relationship via **Correlation-Weighted Edges in ST-GNN v3** further improved accuracy to **94.90%**, confirming that modeling the underlying system topology adds a noticeable advantage.
3.  **Benchmark Dominance:** The 1D-CNN benchmark achieved the highest overall accuracy of **97.82%** on the unseen 3HP load.

---

### Detailed Unseen 3HP Performance (F1-Score)

| Fault Class | 1D-CNN (Mixed) | ST-GNN v1 (Raw Signal) | ST-GNN v2 (Features) | ST-GNN v3 (Features + Weights) |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 1.0000 | 0.9989 | 0.9995 | 0.9984 |
| **Ball Fault** | 0.9549 | 0.7402 | 0.8346 | **0.8891** |
| **Inner Race** | 0.9506 | 0.6033 | 0.9367 | **0.9560** |
| **Outer Race** | **1.0000** | 0.7080 | 0.9026 | 0.9341 |

---

### Tech Stack

The following core libraries were used for data processing, model development, and experimentation:

* **Deep Learning:** PyTorch, PyTorch Geometric (PyG)
* **Experimentation:** Optuna (Hyperparameter Tuning)
* **Data & Scientific Computing:** NumPy, SciPy, Pandas, scikit-learn
