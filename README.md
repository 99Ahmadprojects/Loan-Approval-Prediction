# Loan Approval Prediction ✅

A machine learning project to predict whether a loan application will be **Approved** or **Not Approved** based on applicant financial details, credit history, and asset values.  
This project was built and tested in **Google Colab** using Python, Scikit-learn, and Imbalanced-learn (SMOTE).

---

## 📌 Dataset
The dataset contains the following columns:

- `loan_id`  
- `no_of_dependents`  
- `education`  
- `self_employed`  
- `income_annum`  
- `loan_amount`  
- `loan_term`  
- `cibil_score`  
- `residential_assets_value`  
- `commercial_assets_value`  
- `luxury_assets_value`  
- `bank_asset_value`  
- `loan_status` (Target: Approved / Not Approved)

---

## ⚙️ Features
- **Data Preprocessing**
  - Handled missing values  
  - Encoded categorical features (`education`, `self_employed`)  
  - Scaled numeric features  
- **Class Imbalance Handling**
  - Used **SMOTE** to oversample the minority class  
  - Compared with class-weight balancing  
- **Models Trained**
  - Logistic Regression  
  - Decision Tree  
- **Evaluation Metrics**
  - Precision, Recall, F1-score  
  - ROC-AUC & PR-AUC  
  - Confusion Matrix  

---

## 🛠️ Tools & Libraries
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Matplotlib  

---

## 🚀 Training & Evaluation
1. Mount dataset in Colab and load with Pandas  
2. Apply preprocessing pipeline (numeric + categorical)  
3. Handle class imbalance with SMOTE inside CV pipeline  
4. Train Logistic Regression and Decision Tree using GridSearchCV  
5. Select best model based on **F1-score**  
6. Evaluate on test set with ROC, PR, and confusion matrix  

---

## 📊 Results
- Logistic Regression and Decision Tree compared  
- Best model chosen automatically based on CV F1-score  
- Precision/Recall trade-off tuning with validation split  

---

## 💾 Model Saving
The final trained pipeline (preprocessing + SMOTE + model) is saved with:

```python
import joblib
joblib.dump(best_model, "loan_approval_best_pipeline.joblib")
