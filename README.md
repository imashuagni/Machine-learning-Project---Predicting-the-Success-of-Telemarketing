# Predict the Success of Bank Telemarketing

**MLP Project**  

---

## About this Competition

The data is related to direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required in order to determine if the product (bank term deposit) would be subscribed (`yes`) or not (`no`).

Your goal is to build a model (e.g., a Multilayer Perceptron) to predict whether a client will subscribe to a term deposit based on their demographic and historical interaction data.

---

## My Approach and Solution Method

### Problem Understanding
This is a binary classification problem where I need to predict whether a client will subscribe to a term deposit (target: "yes"/"no") based on 15 input features. The challenge lies in handling mixed data types (numerical and categorical), potential class imbalance, and extracting meaningful insights from customer interaction patterns.

### Data Preprocessing Strategy
I implemented a comprehensive preprocessing pipeline to handle the diverse nature of the dataset:

1. **Date Feature Engineering**: Extracted year, month, and day components from the 'last contact date' to capture temporal patterns in customer behavior.

2. **Feature Categorization**: 
   - Numerical features: age, balance, duration, campaign, pdays, previous, contact_year, contact_month, contact_day
   - Categorical features: job, marital, default, housing, loan, contact
   - Ordinal feature: education (with explicit ranking: primary < secondary < tertiary)

3. **Preprocessing Pipelines**:
   - **Numerical Pipeline**: Mean imputation for missing values + StandardScaler for normalization
   - **Categorical Pipeline**: Most frequent imputation + OneHotEncoder (avoiding dummy variable trap)
   - **Education Pipeline**: Most frequent imputation + OrdinalEncoder with proper ranking

4. **Data Quality**: Removed the 'poutcome' column due to high missing values and potential data leakage concerns.

### Model Selection and Rationale

I implemented and compared four different machine learning models, each chosen for specific strengths:

#### Model 1: Logistic Regression
**Why I chose it:**
- Provides probabilistic outputs for flexible decision-making
- Highly interpretable for business stakeholders
- Computationally efficient and less prone to overfitting
- Excellent baseline model for binary classification

**Results:**
- Test Accuracy: 84.89%
- F1 Score (Macro): 0.57
- ROC AUC: 0.79

#### Model 2: SGD Classifier with SMOTE
**Why I chose it:**
- Handles large datasets efficiently with incremental learning
- Memory efficient for production deployment
- Applied SMOTE to address class imbalance issues

**Key Innovation:** Used SMOTE oversampling to balance the dataset before training.

**Results:**
- Test Accuracy: 80%
- F1 Score (Weighted): 0.82
- ROC AUC: 0.84

#### Model 3: Random Forest (Best Performer)
**Why I chose it:**
- Robust against overfitting through ensemble learning
- Captures non-linear relationships effectively
- Provides feature importance insights
- Handles mixed data types naturally

**Results:**
- Test Accuracy: 86%
- F1 Score (Macro): 0.77
- ROC AUC: Not explicitly calculated
- **Best overall performance**

#### Model 4: Decision Tree
**Why I chose it:**
- Maximum interpretability for business rules
- No feature scaling required
- Handles missing values naturally
- Fast prediction times

**Results:**
- Test Accuracy: 85.21%
- F1 Score (Macro): 0.70
- ROC AUC: 0.84

### Key Insights from Feature Analysis

From the Decision Tree model, I identified the most important predictors:
1. **Duration of last contact** (49.92% importance) - Most critical factor
2. **Days since previous contact** (12.25% importance)
3. **Contact month** (6.62% importance) - Seasonal patterns matter
4. **Housing loan status** (6.59% importance)
5. **Customer age** (6.45% importance)

### Evaluation Strategy

I focused on multiple metrics to ensure robust model evaluation:
- **Accuracy**: Overall prediction correctness
- **F1 Score (Macro)**: Balanced performance across both classes
- **ROC AUC**: Model's ability to distinguish between classes
- **Classification Report**: Detailed precision and recall analysis

### Business Impact Considerations

I designed my evaluation approach with real-world banking implications in mind:

- **Precision Focus**: Minimizing false positives to avoid incorrectly targeting unlikely customers (cost efficiency)
- **Recall Balance**: Ensuring we don't miss potential subscribers (revenue opportunity)
- **F1 Score Optimization**: Balancing both concerns for optimal business outcomes

### Technical Implementation

- Used RandomizedSearchCV and GridSearchCV for hyperparameter optimization
- Implemented proper train/validation/test splits (80/20)
- Applied cross-validation for robust model selection
- Created reproducible results with fixed random seeds
- Generated submission-ready predictions for all models

### Final Model Selection

**Random Forest emerged as the best performer** due to:
- Highest test accuracy (86%)
- Strong F1 macro score (0.77)
- Robust performance without overfitting
- Valuable feature importance insights for business understanding

This approach demonstrates a comprehensive machine learning workflow from data understanding through model deployment, with careful attention to both technical performance and business applicability.

---

## Files

- **train.csv** – The training set  
- **test.csv** – The test set  
- **sample_submission.csv** – A sample submission file in the correct format  

---

## Input Variables

| #  | Variable    | Description                                                                                                                                   |
|----|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | last contact date | Last contact date                                                                                                                       |
| 2  | age         | Age of the client (numeric)                                                                                                                   |
| 3  | job         | Type of job                                                                                                                                   |
| 4  | marital     | Marital status (categorical: "married", "divorced", "single"; note: "divorced" means divorced or widowed)                                     |
| 5  | education   | Education level (categorical: "unknown", "secondary", "primary", "tertiary")                                                                  |
| 6  | default     | Has credit in default? (binary: "yes", "no")                                                                                                  |
| 7  | balance     | Average yearly balance in euros (numeric)                                                                                                     |
| 8  | housing     | Has a housing loan? (binary: "yes", "no")                                                                                                     |
| 9  | loan        | Has a personal loan? (binary: "yes", "no")                                                                                                    |
| 10 | contact     | Contact communication type (categorical: "unknown", "telephone", "cellular")                                                                  |
| 11 | duration    | Last contact duration in seconds (numeric)                                                                                                    |
| 12 | campaign    | Number of contacts performed during this campaign for this client (numeric, includes last contact)                                            |
| 13 | pdays       | Number of days that passed by after the client was last contacted from a previous campaign (numeric; -1 means not previously contacted)      |
| 14 | previous    | Number of contacts performed before this campaign for this client (numeric)                                                                   |
| 15 | poutcome    | Outcome of the previous marketing campaign (categorical: "unknown", "other", "failure", "success")                                            |

---

## Output Variable (Target)

| #  | Variable | Description                                                                                 |
|----|----------|---------------------------------------------------------------------------------------------|
| 16 | target   | Has the client subscribed a term deposit? (binary: "yes", "no")                             |

---

## Getting Started

1. **Explore the data**  
   - Load `train.csv` and examine distributions, missing values, and correlations.

2. **Preprocess features**  
   - Encode categorical variables (one-hot or label encoding).  
   - Scale numerical features if needed (e.g., StandardScaler).

3. **Modeling**  
   - Build a Multilayer Perceptron (MLP) using frameworks such as TensorFlow/Keras or PyTorch.  
   - Experiment with network architecture, activation functions, learning rates, and regularization.

4. **Evaluation**  
   - Use cross-validation on the training set to tune hyperparameters.  
   - Evaluate on a hold-out validation set or via k-folds.  
   - Metrics: accuracy, precision, recall, F1-score, ROC AUC.

5. **Submission**  
   - Train your final model on the full training set.  
   - Generate predictions (`yes`/`no`) on `test.csv`.  
   - Save results in the same format as `sample_submission.csv` and submit.

---
