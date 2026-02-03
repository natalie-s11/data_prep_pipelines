
# %% 
# Imports - Libraries needed for data manipulation and ML preprocessing
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
# Step 1: Review these two datasets and brainstorm problems that could be addressed with the dataset. Identify a question for each dataset.

# Review datasets 
College = pd.read_csv("college_completion.csv")
College.head() 
# Question: Does the state you are from impact if you go to private school?


# %%
Job = pd.read_csv("job_placement.csv")
Job.head()
# Question: Does what you study and concentrate in determine your salary?


# %%
# Step 2: Work through the steps outlined in the examples to include the following elements:
# Write a generic question that this dataset could address.
# Question - College: Does the type of college (2-year/4-year) influence what type of institution it is (public/private)?
# Question - Job: Does gender influence salary?

# What is a independent Business Metric for your problem? Think about the case study examples we have discussed in class.
# IMB College: Does attending an HBCU affect the likelihood of what level of school you attend?
# IMB Job: Does studying science and technology affect job placement?


# %%
# Data preparation:
# View basic information of each dataset
College.info()
Job.info()

# %%
# College 
# correct variable type/class as needed
# change cohort size to fill na with 0 and convert to int
College['cohort_size'] = College['cohort_size'].fillna(0).astype(int)

# Change HBCU column to boolean 
College['hbcu'] = College['hbcu'].apply(lambda x: True if x == 1 else (False if x == 0 else False))

# Change Flagship column to boolean 
College['flagship'] = College['flagship'].apply(lambda x: True if x == 1 else (False if x == 0 else False))

# Create categories for basic and change that column to them.
basic_categories = ['associates', 'masters', 'baccalaureate', 'research', 'other']
College['basic'] = College['basic'].apply(lambda x: True if x in basic_categories else pd.NA)
College['basic'] = College['basic'].astype('boolean')

# Change level to boolean
College['is_four_year'] = College['level'].apply(lambda x: True if x == 'four-year' else (False if x == 'two-year' else pd.NA))
College['is_four_year'] = College['is_four_year'].astype('boolean')

# See Changes
College.dtypes

# Job
# Convert salary to integer, fill na values with
# Note, na values are probably for those not placed
Job['salary'] = Job['salary'].fillna(0).astype(int)
Job['degree_t']

# Change degree_t to categories 
hsc_s_categories = ['Commerce', 'Science', 'Arts']
Job['hsc_s'] = Job['hsc_s'].apply(lambda x: True if x in hsc_s_categories else pd.NA)
Job['hsc_s'] = Job['hsc_s'].astype('boolean')

#Change workex to boolean
Job['workex'] = Job['workex'].apply(lambda x: True if x == 'Yes' else (False if x == 'No' else pd.NA))
Job['workex'] = Job['workex'].astype('boolean')


# %%
# College
# collapse factor levels as needed
# I decided to not do this for College. I was considering it for basic but I do not know if certain research is 2 or 4 years so I chose to not collapse factor levels.

# Job
# I decided to not do this for Job because the dataset is smaller and mostly complete.


# %%
# College
# one-hot encoding factor variables
# Change control to one-hot encode
College = pd.get_dummies(
    College,
    columns=['control'],   
    prefix='control',      
    dummy_na=False         
)
print([col for col in College.columns if 'control' in col])


# Job
# Change degree_t to one-hot encode
Job = pd.get_dummies(
    Job,
    columns=['degree_t'],
    prefix='degree',
    dummy_na=False
)

print([col for col in Job.columns if 'degree_' in col])
# %%
# College
# normalize the continuous variables
# View the range of cohort size.
MaxValue = College['cohort_size'].max()
MinValue = College['cohort_size'].min()
print(MaxValue-MinValue)
# Use the MinMaxScaler() to normalize 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
College['cohort_size_scaled'] = scaler.fit_transform(College[['cohort_size']])

# Do the same for student_count 
MaxValue = College['student_count'].max()
MinValue = College['student_count'].min()
print(MaxValue-MinValue)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
College['student_count'] = scaler.fit_transform(College[['student_count']])

# Job
#Change hsc_p to mix max scale
MaxValue = Job['hsc_p'].max()
MinValue = Job['hsc_p'].min()
print(MaxValue-MinValue)
# Use the MinMaxScaler() to normalize 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
Job['hsc_p'] = scaler.fit_transform(Job[['hsc_p']])


#%%
# College
# drop unneeded variables
# Drop lots of variables
College.drop(['long_x', 'lat_y', 'site', 'awards_per_value', 'awards_per_state_value', 'awards_per_natl_value', 'exp_award_value', 'exp_award_state_value', 'exp_award_natl_value', 'exp_award_percentile', 'ft_pct', 'fte_value', 'fte_percentile', 'vsa_grad_elsewhere_after6_first', 'vsa_enroll_after6_first', 'vsa_enroll_elsewhere_after6_first', 'vsa_grad_after4_transfer', 'vsa_grad_elsewhere_after4_transfer', 'vsa_enroll_after4_transfer', 'vsa_enroll_elsewhere_after4_transfer', 'vsa_grad_after6_transfer', 'vsa_grad_elsewhere_after6_transfer', 'vsa_enroll_after6_transfer', 'vsa_enroll_elsewhere_after6_transfer', 'similar', 'state_sector_ct', 'carnegie_ct', 'counted_pct', 'nicknames', 'vsa_year', 'vsa_grad_after4_first', 'vsa_grad_elsewhere_after4_first', 'vsa_enroll_after4_first', 'vsa_enroll_elsewhere_after4_first', 'vsa_grad_after6_first'], axis=1, inplace=True, errors='ignore')
#Verify columns dropped
College.columns


# Job
#I am going to drop the variables that do not relate to my question
Job.drop(['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_b', 'hsc_p' 'degree_p', 'etest_p', 'mba_p'], axis=1, inplace=True, errors='ignore')
#Verify columns dropped
Job.columns

# %%
# College
# create target variable if needed
# The target variable is the amount of people per state that go to private school.
College['private_school'] = College['control_Private for-profit'] + College['control_Private not-for-profit']
# Calculate private school students per college
College['private_students'] = College['private_school'] * College['cohort_size']
# Group by state and sum to get total private students per state
private_by_state = College.groupby('state')['private_students'].sum().sort_values(ascending=False)
print(private_by_state)

# Job


# %% 
# college
# Calculate the prevalence of the target variable
# Question: Does the state you are from impact if you go to private school?
# Calculate total students per state for percentage
total_by_state = College.groupby('state')['cohort_size'].sum()
pct_private_by_state = (private_by_state / total_by_state * 100).sort_values(ascending=False)
# View results
print(pct_private_by_state)


# Job


# %% 
# College
# Create the necessary data partitions (Train,Tune,Test)
#  Separate training data from the rest for  
train, test = train_test_split(
    College,
    train_size=0.55,                 
    stratify=College['private_school'], 
    random_state=42
)
# Verify the split sizes
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")


# Split remaining data into tuning and test sets
tune, test = train_test_split(
    test,
    train_size=0.5,                  
    stratify=test['private_school'],   
    random_state=42
)

# Verify prevalence in training set
print("Training set class distribution:")
print(train['private_school'].value_counts())
train_prev = train['private_school'].mean()
print(f"Training prevalence: {train_prev:.2%}")

# Verify prevalence in tuning set
print("\nTuning set class distribution:")
print(tune['private_school'].value_counts())
tune_prev = tune['private_school'].mean()
print(f"Tuning prevalence: {tune_prev:.2%}")

# Verify prevalence in test set
print("\nTest set class distribution:")
print(test['private_school'].value_counts())
test_prev = test['private_school'].mean()
print(f"Test prevalence: {test_prev:.2%}")


# Job

# %%
