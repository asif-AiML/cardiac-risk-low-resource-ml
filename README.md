# 🫀 Cardiac Risk Prediction in Low-Resource Environments

**Author:** Muhammad Asif  
**Tech Stack:** Python, Scikit-Learn, Imbalanced-Learn, Pandas, Matplotlib  

## 📌 Project Overview
Machine learning models often achieve high global accuracy on medical datasets by simply ignoring rare diseases. This project tackles the **Class Imbalance Problem** in cardiovascular disease prediction. 

We simulated a "low-resource" or "rare event" scenario using the UCI Cleveland Heart Disease dataset by aggressively reducing positive disease samples, creating a severe **1:7 Imbalance Ratio** (only ~17 positive examples in the training set). 

**The Goal:** Rescue the model's predictive capability and prioritize patient safety (Sensitivity/Recall) over misleading global accuracy.

## 🔬 Methodology: The Hybrid "Turbo" Framework
Standard Logistic Regression failed completely on the sabotaged data. To fix this, we implemented a dual-intervention strategy:

1. **Targeted Synthetic Augmentation (Borderline-SMOTE):** Instead of standard SMOTE, we used Borderline-SMOTE (`k_neighbors=1`) to synthesize data only along the critical decision boundary, preventing the generation of noisy data in an already sparse environment.
2. **Cost-Sensitive Learning & Threshold Tuning:** We applied balanced class weights to heavily penalize False Negatives. Furthermore, we manually shifted the decision threshold from `0.5` to `0.25` to prioritize identifying at-risk patients.

## 📊 Results & Clinical Impact
The baseline model achieved 73% accuracy but missed **66% of sick patients**. Our proposed hybrid method sacrificed a negligible amount of precision to achieve a massive leap in patient safety.

* **Baseline Recall (Sensitivity):** 33.3%
* **Proposed Method Recall:** 75.0% *(More than doubled)*
* **F1-Score Improvement:** 0.50 ➔ 0.84

## 🛠️ How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt` (requires `scikit-learn` and `imbalanced-learn`).
3. Run the Jupyter Notebook located in the `notebooks/` directory.
