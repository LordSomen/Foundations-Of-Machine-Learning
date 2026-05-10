"""
Part 1: Generate Dataset (5 marks)
Generate a synthetic dataset of 2500 employee records. Each record will contain the
following features and a burnout score (target):
1. Work Hours (continuous): Random values between 4 and 14 (hours per day).
2. Job Satisfaction (categorical): Random integers between 1 and 5 (1 = very
dissatisfied, 5 = very satisfied).
3. Overtime (ordinal): Random integers between 0 and 4 (0 = never, 4 = frequent
overtime).
4. Commute Time (continuous): Random values between 0 and 120 (minutes).
The target value (Burnout Score) should be generated using the following formula:
Burnout Score = 0.1×Work Hours − 0.15×Job Satisfaction + 0.2×Overtime +
0.05×Commute Time + Noise
For Noise, Draw random values from a normal distribution with mean = 0 and std = 0.1.
After calculating the burnout scores, normalize them to lie between 0 and 1. Make sure
that the data is not imbalanced. Split the data into train : val : test = 70 : 15 : 15 sets
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Number of samples
n_samples = 2500

# Generate features
work_hours = np.random.uniform(4, 14, n_samples)
job_satisfaction = np.random.randint(1, 6, n_samples)
overtime = np.random.randint(0, 5, n_samples)
commute_time = np.random.uniform(0, 120, n_samples)

# Generate Noise
noise = np.random.normal(0, 0.1, n_samples)

# Calculate Burnout Score
burnout_score = (
    0.1 * work_hours
    - 0.15 * job_satisfaction
    + 0.2 * overtime
    + 0.05 * commute_time
    + noise
)

# Normalize Burnout Score
burnout_score = (burnout_score - burnout_score.min()) / (burnout_score.max() - burnout_score.min())

# Create DataFrame
data = pd.DataFrame({
    'Work Hours': work_hours,
    'Job Satisfaction': job_satisfaction,
    'Overtime': overtime,
    'Commute Time': commute_time,
    'Burnout Score': burnout_score
})

# Split the data
train_data, temp_data = train_test_split(data, test_size=0.30, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)



# Verify the splits
print(f"Train set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

try:
    # Save train_data to a CSV file
    train_data.to_csv('./Employee_Burnout/train_data.csv', index=False)

    # Save validation data to a CSV file
    val_data.to_csv('./Employee_Burnout/validation_data.csv', index=False)

    # Save test data to a CSV file
    test_data.to_csv('./Employee_Burnout/test_data.csv', index=False)
    print("successfully saved the datasets!!!")
except Exception as e:
    print("falied to save the dataset, ERROR",e)