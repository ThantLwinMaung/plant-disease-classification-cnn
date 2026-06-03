#  Plant Disease Classification using CNN
##  Project Overview

Plant diseases can significantly reduce crop yields and threaten agricultural productivity and food security. Early and accurate disease identification enables farmers to take timely action, reduce crop losses, and improve agricultural sustainability.

This project develops a **Convolutional Neural Network (CNN)** model to automatically classify plant diseases from leaf images using the **PlantVillage dataset**. The project demonstrates the application of deep learning and computer vision techniques in precision agriculture.

---

##  Dataset
- **Dataset:** PlantVillage
- **Total Classes:** 38
- **Image Type:** Plant Leaf Images
- **Categories:** Healthy and Diseased Crop Leaves

The PlantVillage dataset contains images from multiple crop species and disease conditions, making it a widely used benchmark dataset for agricultural image classification research.

### Dataset Source

This project uses the **PlantVillage dataset** obtained from Kaggle.

**Dataset Link:** https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

The dataset contains approximately **54,000+ images** of healthy and diseased plant leaves across **38 classes** and is widely used for agricultural image classification research.

---

##  Project Workflow

1. Data Loading and Exploration
2. Image Preprocessing and Normalization
3. CNN Model Development
4. Model Training
5. Performance Evaluation
6. Disease Prediction on New Images

---

##  Model Architecture

The CNN architecture consists of:

- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D
- Flatten Layer
- Dense Layer (128 Neurons)
- Output Layer (38 Classes)

---

##  Results

| Metric | Value |
|---------|---------|
| Validation Accuracy | 88.23% |
| Number of Classes | 38 |
| Confusion Matrix | Generated |
| Classification Report | Generated |

---

##  Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

##  Future Improvements

- Data Augmentation
- Early Stopping and Model Checkpointing
- Transfer Learning (MobileNetV2, EfficientNet)
- Streamlit Deployment
- Explainable AI using Grad-CAM
- Testing on Real-World Field Images

---

##  Motivation

This project combines agriculture and artificial intelligence by applying deep learning techniques to crop disease detection. It demonstrates how computer vision can support:

- Precision Agriculture
- Disease Monitoring
- Crop Management
- Sustainable Farming Practices

As an agriculture professional transitioning into AI and Data Science, this project represents an important step toward applying machine learning and computer vision to real-world agricultural challenges.

---

##  Acknowledgments

- PlantVillage Dataset Contributors
- Kaggle for hosting and distributing the dataset
- TensorFlow and Keras communities
- Open-source machine learning community
