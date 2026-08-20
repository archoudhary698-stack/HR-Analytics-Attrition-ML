# HR Analytics: Employee Attrition Prediction & Compensation Modeling

End-to-end **supervised + unsupervised** machine learning case study on the IBM HR Employee Attrition dataset (1,470 employees, 35 features).

## Business problem

Replacing an employee often costs 20–30% of annual salary. This project answers two questions HR actually asks:

1. **Classification:** Which employees are likely to leave? (`Attrition`)
2. **Regression:** What monthly compensation is consistent with role, experience and performance? (`MonthlyIncome`)

A third, unsupervised question is used for exploration:

3. **Clustering / PCA:** Do employees form natural segments (for example high-earning stable vs overtime-heavy junior roles)?

## What this project demonstrates

| Topic | Where it appears |
|---|---|
| Supervised vs Unsupervised | Separate pipelines + PCA / K-Means exploration |
| Bias–variance tradeoff | Polynomial degrees, Decision Tree depth, learning curves, Ridge vs Lasso |
| Linear, Ridge, Lasso, Polynomial Regression | Compensation model |
| Logistic Regression, Decision Tree, Random Forest, SVM, KNN | Attrition model |
| Scaling, encoding, selection, transformation | Feature engineering section |
| Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix | Model evaluation dashboard |

## Dataset

IBM HR Analytics Employee Attrition & Performance (fictional data created by IBM).  
Loaded in the notebook from a public GitHub raw URL — .

## How to run

Open `HR_Analytics_Complete_Project.ipynb` in [Google Colab](https://colab.research.google.com) and click **Runtime → Run all**.  


## Tech

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn
