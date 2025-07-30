# 🚀 Income Prediction for the Newland Mission (2048)

This supervised machine learning project was developed in the context of an academic course and addresses a futuristic scenario where Earth became uninhabitable and humans began migrating to a new planet, **Newland**. The objective was to build a model capable of classifying citizens according to their income, helping the Newland government define fair taxation rules based on predicted income brackets.

---

## 🧠 Project Objectives

- Build a **binary classification model** to predict whether a citizen's income is **above** or **below/equal to** the average.
- Optimize the model to maximize **F1 Score**, the official evaluation metric of the competition.
- Contribute actionable insights for **policy and taxation** decisions on Newland using interpretable, robust ML techniques.

---

## 📊 Dataset Description

Two datasets were provided:

- **Training set (22,400 records)**: Contains features and income class labels (0 = below/equal to average; 1 = above average).
- **Test set (10,100 records)**: Features only; our task was to predict the income class.

### Key Attributes:

| Variable                     | Description                                                    |
|-----------------------------|----------------------------------------------------------------|
| `Citizen ID`                | Unique identifier                                              |
| `Birthday`                  | Date of birth                                                 |
| `Native Continent`          | Origin on Earth                                               |
| `Marital Status`            | Marital status                                                |
| `Lives with`                | Household structure                                           |
| `Base Area`                 | Assigned neighborhood on Newland                             |
| `Education Level`           | Highest academic qualification                                |
| `Years of Education`        | Duration of formal education                                  |
| `Employment Sector`         | Work industry                                                 |
| `Role`                      | Job function                                                  |
| `Working Hours per Week`    | Weekly work hours                                             |
| `Money Received`            | Subsidy received by Group B                                   |
| `Ticket Price`              | Payment made by Group C participants                          |
| `Income` (train only)       | Target variable (0 or 1)                                      |

---

## ⚙️ Methodology

### 1. 🧼 Preprocessing
- Cleaned invalid and missing values using **KNN Imputer**
- Extracted age from `Birthday`
- One-hot encoded categorical variables
- Normalized numerical variables using **Min-Max Scaling**

### 2. 📊 Exploratory Data Analysis
- Visualized class balance, sector distributions, and education levels
- Identified key correlations between income and features like:
  - **Working hours**
  - **Years of education**
  - **Base area**
  - **Ticket Price / Money Received**

### 3. 🧠 Modeling Approaches
We trained and compared several models:

| Model                  | Notes                                 |
|------------------------|----------------------------------------|
| Logistic Regression    | Baseline model                        |
| Random Forest          | Strong performer, good interpretability |
| Gradient Boosting      | Tuned using `RandomizedSearchCV`       |
| Neural Network (MLP)   | High complexity, regularized           |
| k-NN                   | Used for comparison and imputation     |

> We also tested **PCA** to evaluate dimensionality reduction, but decided not to use it in the final pipeline.

### 4. 🧪 Model Evaluation
- Used **K-Fold cross-validation (k=5)**
- Measured:
  - Accuracy
  - Precision, Recall
  - F1 Score (primary metric)
  - ROC AUC
- Final model selected based on **validation F1** and **generalization capacity** to test set

---

## 📈 Results

- The best model (Random Forest + tuned features) achieved:
  - **F1 Score:** 0.71201
  - 🏆 **Ranked 8th out of 39 teams**
- Final model prioritized **recall** to avoid misclassifying high-income earners into the lower-tax group
  
---

## 🔎 Feature Importance

Most influential variables in predicting income class:

1. `Working Hours per Week`
2. `Years of Education`
3. `Role` (encoded)
4. `Base Area`
5. `Money Received` / `Ticket Price`

---

## 📦 Tools & Libraries

- **Python 3.10**
- **Google Colab**
- pandas, numpy, matplotlib, seaborn
- scikit-learn: modeling, imputation, pipelines
- plotly (EDA)
- scipy (feature selection)
