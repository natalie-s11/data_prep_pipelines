# Step 4: Create functions for your two pipelines that produces the train and test datasets.

# %%
# Imports
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
import requests  # For HTTP requests to download data


# %% 
# Re-load datasets in
College = pd.read_csv("college_completion.csv")
Job = pd.read_csv("job_placement.csv")

# %%
# Questions:
# College - Does the state you are from impact if you go to private school?
# Job - Does what you study and concentrate in determine your salary?

# %%
# College pipeline:
# Define pipeline functions for College
def prep_college_pipeline(college_path):
    College = pd.read_csv(college_path)

    # Change data types
    College['cohort_size'] = College['cohort_size'].fillna(0).astype(int)

    College['hbcu'] = College['hbcu'].apply(lambda x: True if x == 1 else False)
    College['flagship'] = College['flagship'].apply(lambda x: True if x == 1 else False)

    College['is_four_year'] = College['level'].apply(
        lambda x: True if x == 'four-year' else (False if x == 'two-year' else pd.NA)
    ).astype('boolean')

    # One-hot encode 
    College = pd.get_dummies(College, columns=['control'], prefix='control')

    # Scaling 
    scaler = MinMaxScaler()
    College['cohort_size_scaled'] = scaler.fit_transform(College[['cohort_size']])
    College['student_count_scaled'] = scaler.fit_transform(College[['student_count']])

    # Drop columns
    College.drop([
        'long_x', 'lat_y', 'site', 'awards_per_value', 'awards_per_state_value',
        'awards_per_natl_value', 'exp_award_value', 'exp_award_state_value',
        'exp_award_natl_value', 'exp_award_percentile', 'ft_pct', 'fte_value',
        'fte_percentile', 'vsa_grad_elsewhere_after6_first', 'vsa_enroll_after6_first',
        'vsa_enroll_elsewhere_after6_first', 'vsa_grad_after4_transfer',
        'vsa_grad_elsewhere_after4_transfer', 'vsa_enroll_after4_transfer',
        'vsa_enroll_elsewhere_after4_transfer', 'vsa_grad_after6_transfer',
        'vsa_grad_elsewhere_after6_transfer', 'vsa_enroll_after6_transfer',
        'vsa_enroll_elsewhere_after6_transfer', 'similar', 'state_sector_ct',
        'carnegie_ct', 'counted_pct', 'nicknames', 'vsa_year',
        'vsa_grad_after4_first', 'vsa_grad_elsewhere_after4_first',
        'vsa_enroll_after4_first', 'vsa_enroll_elsewhere_after4_first',
        'vsa_grad_after6_first'
    ], axis=1, inplace=True, errors='ignore')

    # Define the target variable
    College['private_school'] = (
        College['control_Private for-profit'] +
        College['control_Private not-for-profit']
    )

    # Train / Tune / Test Split
    train, temp = train_test_split(
        College,
        train_size=0.55,
        stratify=College['private_school'],
        random_state=42
    )

    tune, test = train_test_split(
        temp,
        train_size=0.5,
        stratify=temp['private_school'],
        random_state=42
    )

    # Specify what to return
    return train, tune, test


# %%
# Job pipeline:
# Define pipeline functions for College
def prep_job_pipeline(job_path):
    Job = pd.read_csv(job_path)

    # Change data types
    Job['salary'] = Job['salary'].fillna(0).astype(int)

    Job['workex'] = Job['workex'].apply(lambda x: True if x == 'Yes' else False).astype('boolean')
    Job['status'] = Job['status'].apply(lambda x: True if x == 'Yes' else False).astype('boolean')

    # One-hot encode 
    Job = pd.get_dummies(Job, columns=['degree_t'], prefix='degree')
    Job = pd.get_dummies(Job, columns=['specialisation'], prefix='spec')

    # Rename columns if necessary
    Job = Job.rename(columns={'degree_p': 'degree_pct'})

    # Scaling
    scaler = MinMaxScaler()
    Job['hsc_p_scaled'] = scaler.fit_transform(Job[['hsc_p']])

    # Drop columns
    Job.drop([
        'sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_b',
        'etest_p', 'mba_p'
    ], axis=1, inplace=True, errors='ignore')

    # Drop rows with missing values
    Job = Job.dropna(subset=['salary'])

    # Define the target variable
    Job['salary_group'] = pd.qcut(Job['salary'], q=3, labels=['Low', 'Medium', 'High'])

    # Train / Tune / Test Split 
    train, temp = train_test_split(
        Job,
        train_size=0.55,
        stratify=Job['salary_group'],
        random_state=42
    )

    tune, test = train_test_split(
        temp,
        train_size=0.5,
        stratify=temp['salary_group'],
        random_state=42
    )

    # Specify what to return
    return train, tune, test



#%%
# Run College pipeline
college_train, college_tune, college_test = prep_college_pipeline("college_completion.csv")


#%%
# Run Job pipeline
job_train, job_tune, job_test = prep_job_pipeline("job_placement.csv")


#%%
# Confirm the splits based on the question initially asked
print("College splits:")
print(college_train.shape, college_tune.shape, college_test.shape)
print(college_train['private_school'].value_counts(normalize=True))

print("\nJob splits:")
print(job_train.shape, job_tune.shape, job_test.shape)
print(job_train['salary_group'].value_counts(normalize=True))