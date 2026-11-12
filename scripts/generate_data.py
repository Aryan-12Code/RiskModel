"""
Script to generate synthetic credit/financial risk dataset.
Creates a realistic dataset with ~5000 rows and ~20% default rate.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 5000

# Base features
age = np.random.normal(40, 12, n).clip(18, 75).astype(int)
income = np.random.lognormal(10.8, 0.6, n).clip(15000, 500000).round(2)
employment_length = np.random.exponential(5, n).clip(0, 35).round(1)
credit_score = np.random.normal(680, 80, n).clip(300, 850).astype(int)

# Loan features correlated with income
loan_amount = (income * np.random.uniform(0.2, 2.5, n)).clip(1000, 500000).round(2)
# Interest rate inversely correlated with credit score
interest_rate = (25 - credit_score / 50 + np.random.normal(0, 2, n)).clip(2.5, 30.0).round(2)

# Debt metrics
debt_to_income = (np.random.beta(2, 5, n) * 80).clip(1, 80).round(2)
num_credit_lines = np.random.poisson(5, n).clip(0, 30)
num_late_payments = np.random.exponential(1.5, n).clip(0, 20).astype(int)

# Categorical features
loan_purpose = np.random.choice(
    ['personal', 'education', 'medical', 'business', 'home_improvement', 'debt_consolidation'],
    n, p=[0.25, 0.15, 0.10, 0.15, 0.15, 0.20]
)
home_ownership = np.random.choice(['own', 'rent', 'mortgage'], n, p=[0.3, 0.35, 0.35])

# Financial features
annual_revenue = (income * np.random.uniform(0.8, 1.4, n)).round(2)
monthly_expenses = (income / 12 * np.random.uniform(0.4, 0.95, n)).round(2)
savings_balance = (income * np.random.exponential(0.5, n)).clip(0, 500000).round(2)

# Generate default target with realistic correlations
# Higher default probability for: high DTI, low credit score, many late payments, low income
logit = (
    -3.0
    + 0.04 * (debt_to_income - 25)
    - 0.008 * (credit_score - 680)
    + 0.15 * num_late_payments
    - 0.000005 * (income - 60000)
    + 0.02 * (interest_rate - 12)
    + 0.3 * (home_ownership == 'rent').astype(float)
    - 0.2 * (loan_purpose == 'home_improvement').astype(float)
    + 0.25 * (loan_purpose == 'debt_consolidation').astype(float)
)
prob_default = 1 / (1 + np.exp(-logit))
# Scale to target ~20% default rate
prob_default = prob_default * (0.20 / prob_default.mean())
prob_default = prob_default.clip(0.01, 0.95)
default = (np.random.random(n) < prob_default).astype(int)

df = pd.DataFrame({
    'age': age,
    'income': income,
    'employment_length': employment_length,
    'loan_amount': loan_amount,
    'interest_rate': interest_rate,
    'credit_score': credit_score,
    'debt_to_income': debt_to_income,
    'num_credit_lines': num_credit_lines,
    'num_late_payments': num_late_payments,
    'loan_purpose': loan_purpose,
    'home_ownership': home_ownership,
    'annual_revenue': annual_revenue,
    'monthly_expenses': monthly_expenses,
    'savings_balance': savings_balance,
    'default': default
})

# Inject ~2% missing values in select columns for realism
for col in ['income', 'employment_length', 'credit_score', 'savings_balance', 'annual_revenue']:
    mask = np.random.random(n) < 0.02
    df.loc[mask, col] = np.nan

output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_credit_data.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Generated dataset: {len(df)} rows, default rate: {df['default'].mean():.2%}")
print(f"Saved to {output_path}")
