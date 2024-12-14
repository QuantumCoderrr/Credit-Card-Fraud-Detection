# Credit Card Fraud Detection 🚨💳

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It uses a dataset containing credit card transactions, where each transaction is labeled as either 'Fraud' or 'Not Fraud'. The goal is to train a model to predict fraud based on transaction features.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
In this project, we use a **Random Forest Classifier** to classify credit card transactions as fraudulent or not. The project includes steps like:
- Data Preprocessing
- Model Training
- Evaluation (Confusion Matrix, Classification Report, ROC-AUC)
- Feature Importance Analysis

## Requirements 📦
The following libraries are required to run this project:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Getting Started 🚀

### Prerequisites
- Python 3.8 or higher
- Required libraries installed (`pip install -r requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/QuantumCoderrr/CreditCardFraudDetection.git
   cd CreditCardFraudDetection
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
## Results 📊
Below are the visuals showing the **Confusion Matrix** and **Feature Importance**.

### Confusion Matrix
The confusion matrix visualizes the performance of the classification model, showing the true positives, true negatives, false positives, and false negatives.

![Confusion Matrix](images/confusion_matrix.png)

### Feature Importance
Feature importance indicates the relative importance of each feature in the model's decision-making process. Higher values indicate features that play a greater role in determining the prediction.

![Feature Importance](images/feature_importance.png)

## Dataset 📂
The dataset used for this project can be accessed via the following Google Drive link:  
[Credit Card Fraud Detection Dataset](https://drive.google.com/file/d/15ky1Zn1BTtSR2GCHJCPSVEMtmIO4wXPl/view?usp=drive_link)

## Contributing 🤝
We welcome contributions! Please follow the [contributing guidelines](CONTRIBUTING.md) to submit changes.

## License 📝
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thanks for checking out the project! Let's work together to make fraud detection more efficient! 🚀

