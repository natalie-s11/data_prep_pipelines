#%% 
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


#%% 
# Series of Functions for College

def load_college(path):
    return pd.read_csv(path)

# Change data types
def clean_college_types(df):
    df['cohort_size'] = df['cohort_size'].fillna(0).astype(int)
    df['hbcu'] = df['hbcu'].apply(lambda x: True if x == 1 else False)
    df['flagship'] = df['flagship'].apply(lambda x: True if x == 1 else False)
    df['is_four_year'] = df['level'].apply(
        lambda x: True if x == 'four-year' else (False if x == 'two-year' else pd.NA)
    ).astype('boolean')
    return df

# One-hot encode
def encode_college(df):
    df = pd.get_dummies(df, columns=['control'], prefix='control')
    return df

# Scale 
def scale_college(df):
    scaler = MinMaxScaler()
    df['cohort_size_scaled'] = scaler.fit_transform(df[['cohort_size']])
    df['student_count_scaled'] = scaler.fit_transform(df[['student_count']])
    return df

# Drop columns 
def drop_college_columns(df):
    df = df.drop([
        'long_x','lat_y','site','awards_per_value','awards_per_state_value',
        'awards_per_natl_value','exp_award_value','exp_award_state_value',
        'exp_award_natl_value','exp_award_percentile','ft_pct','fte_value',
        'fte_percentile','similar','nicknames'
    ], axis=1, errors='ignore')
    return df

# Create target variable
def create_college_target(df):
    df['private_school'] = (
        df['control_Private for-profit'] +
        df['control_Private not-for-profit']
    )
    return df

# Train, tune, test split
def split_college(df):
    train, temp = train_test_split(
        df,
        train_size=0.55,
        stratify=df['private_school'],
        random_state=42
    )
    tune, test = train_test_split(
        temp,
        train_size=0.5,
        stratify=temp['private_school'],
        random_state=42
    )
    return train, tune, test

# Put all the functions into one final function
def college_pipeline(path):
    df = load_college(path)
    df = clean_college_types(df)
    df = encode_college(df)
    df = scale_college(df)
    df = drop_college_columns(df)
    df = create_college_target(df)
    return split_college(df)


# %%
# Run the College Pipeline
college_train, college_tune, college_test = college_pipeline("college_completion.csv")

#%% 
#Confirm the splits for College
print("College splits:")
print(college_train.shape, college_tune.shape, college_test.shape)

print("\nCollege class distribution (train):")
print(college_train['private_school'].value_counts(normalize=True))

print("\nCollege class distribution (tune):")
print(college_tune['private_school'].value_counts(normalize=True))

print("\nCollege class distribution (test):")
print(college_test['private_school'].value_counts(normalize=True))

#%%
# Series of Functions for Job
def load_job(path):
    return pd.read_csv(path)

# Change data types
def clean_job_types(df):
    df['salary'] = df['salary'].fillna(0).astype(int)
    df['workex'] = df['workex'].apply(lambda x: True if x == 'Yes' else False)
    df['status'] = df['status'].apply(lambda x: True if x == 'Yes' else False)
    return df

# One hot encode
def encode_job(df):
    df = pd.get_dummies(df, columns=['degree_t'], prefix='degree')
    df = pd.get_dummies(df, columns=['specialisation'], prefix='spec')
    df = df.rename(columns={'degree_p': 'degree_pct'})
    return df

# Scale
def scale_job(df):
    scaler = MinMaxScaler()
    df['hsc_p_scaled'] = scaler.fit_transform(df[['hsc_p']])
    return df

# Create target variable
def create_job_target(df):
    df = df.dropna(subset=['salary'])
    df['salary_group'] = pd.qcut(df['salary'], q=3, labels=['Low', 'Medium', 'High'])
    return df

# Train, tune, test split
def split_job(df):
    train, temp = train_test_split(
        df,
        train_size=0.55,
        stratify=df['salary_group'],
        random_state=42
    )
    tune, test = train_test_split(
        temp,
        train_size=0.5,
        stratify=temp['salary_group'],
        random_state=42
    )
    return train, tune, test

# Put all the functions into one final function
def job_pipeline(path):
    df = load_job(path)
    df = clean_job_types(df)
    df = encode_job(df)
    df = scale_job(df)
    df = create_job_target(df)
    return split_job(df)


# %%
# Run the Job Pipeline
job_train, job_tune, job_test = job_pipeline("job_placement.csv")


#%% 
# Confirm the splits for Job
print("\nJob splits:")
print(job_train.shape, job_tune.shape, job_test.shape)

print("\nJob salary group distribution (train):")
print(job_train['salary_group'].value_counts(normalize=True))

print("\nJob salary group distribution (tune):")
print(job_tune['salary_group'].value_counts(normalize=True))

print("\nJob salary group distribution (test):")
print(job_test['salary_group'].value_counts(normalize=True))



# %%
# College Pipelines as one long function
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
# Job Pipelines as one long function
# Define pipeline functions for Job

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
# %%
