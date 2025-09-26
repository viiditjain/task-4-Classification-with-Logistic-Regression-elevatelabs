# task-4-Classification-with-Logistic-Regression-elevatelabs

# Logistic Regression - Binary Classification Task

## 📌 Objective
The goal of this task is to build a **binary classifier using Logistic Regression**.  
We are using the **Breast Cancer Wisconsin Dataset** to predict whether a tumor is **Malignant (cancerous)** or **Benign (non-cancerous)**.

---

## 🛠 Tools & Libraries
- Python  
- Pandas (data handling)  
- NumPy (mathematical operations)  
- Scikit-learn (ML model & evaluation)  
- Matplotlib & Seaborn (data visualization)  

---

## 📊 Dataset
- **Dataset Used:** Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Target Column:** `diagnosis`  
  - `M` → Malignant → Encoded as `1`  
  - `B` → Benign → Encoded as `0`  
- **Features:** 30 numerical features (radius, texture, smoothness, area, etc.)  

---

## 🔎 Step-by-Step Workflow

### **1. Data Loading & Exploration**
- Loaded dataset with Pandas  
- Checked null values, data types, and basic statistics  

### **2. Data Preprocessing**
- Removed unnecessary columns (`id`, `Unnamed: 32`)  
- Converted categorical labels (M/B) to numerical (1/0)  
- Split data into **features (X)** and **target (y)**  
- Performed **train/test split** (80% train, 20% test)  
- Applied **standardization (scaling)** to normalize feature values  

### **3. Model Training**
- Used **Logistic Regression** from scikit-learn  
- Trained model on training data  

### **4. Model Evaluation**
- Predictions made on test set  
- Evaluation metrics used:
  - **Confusion Matrix**  
  - **Precision, Recall, F1-score**  
  - **ROC Curve**  
  - **ROC-AUC Score**  

### **5. Threshold Tuning**
- Adjusted decision threshold from default `0.5`  
- Observed changes in confusion matrix and metrics  

---

## 📈 Results
- Logistic Regression performed well on the dataset  
- Key metrics (example, replace with your actual results after running code):  
  - Accuracy: ~95%  
  - Precision: ~94%  
  - Recall: ~96%  
  - ROC-AUC Score: ~0.98  

---

## 🧠 Key Concepts Explained

### 🔹 Logistic Regression vs Linear Regression
- **Linear Regression** → Predicts continuous values  
- **Logistic Regression** → Predicts probabilities for classification problems (0 or 1)  

### 🔹 Sigmoid Function
- Formula: `σ(z) = 1 / (1 + e^(-z))`  
- Maps any value into range **0 to 1**  
- Helps convert linear output into probability  

### 🔹 Confusion Matrix
|               | Predicted Positive | Predicted Negative |
|---------------|---------------------|---------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### 🔹 Precision vs Recall
- **Precision** = TP / (TP + FP) → Of all predicted positives, how many are correct  
- **Recall** = TP / (TP + FN) → Of all actual positives, how many were detected  

### 🔹 ROC-AUC
- **ROC Curve** → Plot of TPR vs FPR at different thresholds  
- **AUC (Area Under Curve)** → Higher is better (closer to 1)  

---

