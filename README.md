# Churn Prediction Using Neural Networks 

## Overview 

The project builds a churn prediction model using neural networks trained on customer data. The goal is to identify **potential churners**, helping business reduce revenue loss. 

## Business Objective

- Our goal is maximizing recall on churners, ensuring we are catching actual churners as possible.


## Methodology 

- **Model**: Neural Networks 
    ```python 
    ChurnNN(
        (layer1): Linear(in_features=11, out_features=128, bias=True)
        (layer2): Linear(in_features=128, out_features=64, bias=True)
        (layer3): Linear(in_features=64, out_features=1, bias=True)
        (relu): ReLU()
        (dropout): Dropout(p=0.15, inplace=False)
        )
    ```
- **Loss Function**: `BCEWithLogitsLoss` with positive class weighting for class imbalance
- **Optimizer**: Adam, learning rate = 0.001
- **Threshold**: Default 0.5, tunable for business trade-offs
- **Evaluation Metrics**: Accuracy, Recall, F1 Score


## Key Results 

- **Best Epoch** 40. 

    Overfitting begins when epoch is at the 100+ range.
- **Final Retrain**: Using the best epoch and evaluated on the test set.



## Business Impact Evaluation 

### Test Set performance:

- **ACCURACY** SCORE: 73%
- **F1 SCORE** (churn = 1): 61%
- **RECALL** (churn = 1): 81%


**Estimated Retained Customers Value** 
Average Monthly Charge of Churners: $74.44 USD

***Retained Monthly Revenue***: 
81 users x $74.44 USD = **$6029.64 per 100 customers** 

**False Positives**: 51% of flagged churners may not churn (review potential retention costs)