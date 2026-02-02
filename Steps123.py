
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
# Question: Does where you are from impact where you go and if it is public or private school?

Job = pd.read_csv("job_placement.csv")
Job.head()
# Question: Does what you study and concentrate in determine your salary?
# %%
# Step 2: Work through the steps outlined in the examples to include the following elements:
# Write a generic question that this dataset could address.
# Question - College: Does the type of college influence what type of institution it is?
# Question - Job: Does gender influence salary?

# What is a independent Business Metric for your problem? Think about the case study examples we have discussed in class.
# IMB College: Does attending an HBCU affect the likelihood of transferring?
# IMB Job: Does studying science and technology affect job placement?


# %%
# Data preparation:
# correct variable type/class as needed
# collapse factor levels as needed
# one-hot encoding factor variables
# normalize the continuous variables
# drop unneeded variables
# create target variable if needed


# %% 
# Calculate the prevalence of the target variable
# Create the necessary data partitions (Train,Tune,Test)
