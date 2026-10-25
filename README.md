# Exploratory Data Analysis (EDA): Medical Insurance Cost Prediction

This repository contains an **Exploratory Data Analysis (EDA)** performed on the **Medical Cost Personal Datasets** from Kaggle. The goal of this analysis is to uncover key factors influencing individual medical healthcare expenses and prepare the data for downstream machine learning regression models.

---

## 📌 Dataset Overview

- **Source:** [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **File:** `insurance.csv`
- **Total Records:** 1,338 rows
- **Total Features:** 7 columns

### Features Description

| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Integer | Age of primary beneficiary (18 to 64 years) |
| `sex` | Categorical | Insurance contractor gender (`female`, `male`) |
| `bmi` | Float | Body mass index ($kg / m^2$), measuring body weight relative to height |
| `children` | Integer | Number of children/dependents covered by health insurance (0 to 5) |
| `smoker` | Categorical | Smoking status (`yes`, `no`) |
| `region` | Categorical | Beneficiary's residential area in the US (`northeast`, `southeast`, `southwest`, `northwest`) |
| **`charges`** | Float | Individual medical costs billed by health insurance in USD (**Target Variable**) |

---

## 🔍 Key Findings & EDA Summary

1. **Smoking is the Primary Driver of Medical Costs:**
   - Beneficiaries who smoke have an average medical bill of **$32,050**, compared to **$8,440** for non-smokers (nearly a **4x increase**).
   - Smoking status is by far the single most significant predictor of high charges.

2. **BMI × Smoking Interaction Effect:**
   - For non-smokers, higher BMI results in only slight increases in medical costs.
   - For smokers, a BMI exceeding the clinical obesity threshold ($BMI \ge 30$) causes medical charges to surge drastically past **$35,000 to $40,000+**.

3. **Age Progression:**
   - Medical charges demonstrate a consistent positive linear relationship with age ($r \approx 0.299$) across all demographic groups.

4. **Target Variable Distribution:**
   - The target variable `charges` is positively skewed (skewness = 1.51) with a heavy right tail of high expenses. Applying a logarithmic transformation ($\log(1 + y)$) is recommended when fitting linear regression models.

5. **Minor Predictors:**
   - Gender and geographic region exhibit relatively minimal direct variance in charges, although the Southeast region displays slightly higher average costs due to higher regional smoking and BMI rates.

---

## 🛠️ Data Preparation for Model Training

The notebook demonstrates the necessary preprocessing steps to prepare the dataset for predictive machine learning models (e.g., Linear Regression, Ridge, Random Forest, XGBoost):

- **Binary Encoding:**
  - `sex`: `female` $\rightarrow 0$, `male` $\rightarrow 1$
  - `smoker`: `no` $\rightarrow 0$, `yes` $\rightarrow 1$
- **One-Hot Encoding:**
  - `region`: Transformed using `pd.get_dummies(..., drop_first=True)` to avoid multicollinearity.
- **Train-Test Split:**
  - 80% Training set (`X_train`: 1,069 samples)
  - 20% Testing set (`X_test`: 268 samples)

---

## 📁 Repository Structure

```text
├── insurance.csv               # Raw dataset from Kaggle
├── medical_insurance_eda.ipynb # Complete Jupyter Notebook with code, outputs, and visualizations
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # Project documentation and summary
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/rishbrawat/medical-insurance-cost-eda.git
cd medical-insurance-cost-eda
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook medical_insurance_eda.ipynb
```
