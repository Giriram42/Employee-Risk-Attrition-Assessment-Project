# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
#Read the analytics csv file and store our dataset into a dataframe called "df"
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler
from flask import Flask, render_template, request
df = pd.read_csv("Employee data.csv", index_col=None)

# Renaming certain columns for better readability
df = df.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })

# Convert these variables into categorical variables
df["department"] = df["department"].astype('category').cat.codes
df["salary"] = df["salary"].astype('category').cat.codes


# Move the reponse variable "turnover" to the front of the table
front = df['turnover']
df.drop(labels=['turnover'], axis=1,inplace = True)
df.insert(0, 'turnover', front)

# Create an intercept term for the logistic regression equation
df['int'] = 1
indep_var = ['satisfaction', 'evaluation', 'yearsAtCompany', 'int', 'turnover']
df = df[indep_var]

# Create train and test splits
target_name = 'turnover'
X = df.drop('turnover', axis=1)

y=df[target_name]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)

import statsmodels.api as sm
iv = ['satisfaction','evaluation','yearsAtCompany', 'int']
logReg = sm.Logit(y_train, X_train[iv])
answer = logReg.fit()


# Your function to calculate retention percentage
coef = answer.params
def calculate_retention(coef, Satisfaction, Evaluation, YearsAtCompany) : 
    y1 =  coef[3] + coef[0]*Satisfaction + coef[1]*Evaluation + coef[2]*YearsAtCompany
    p = np.exp(y1) / (1+np.exp(y1))
    return f'An Employee with {Satisfaction} Satisfaction and {Evaluation} Evaluation and worked {YearsAtCompany} years has a {p*100:.2f}% chance of turnover.'

app = Flask(__name__)


# Define your route for the homepage
@app.route('/home')
def index():
    return render_template('index.html')

# Define route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form or request
    satisfaction = float(request.form['satisfaction'])
    evaluation = float(request.form['evaluation'])
    years = float(request.form['years'])

    # Calculate retention percentage using your function
    retention_result = calculate_retention(coef,satisfaction, evaluation, years)

    return render_template('result.html', retention_result=retention_result)

if __name__ == '__main__':
    app.run(debug=True)