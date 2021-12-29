import mlflow
import pandas as pd

logged_model = './catboost-model'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

age = 59
job = 'admin.'
marital = 'married'
education = 'secondary'
default = 'no'
balance = '2343'
housing = 'yes'
loan = 'no'
contact = 'unknown'
day = 5
month = 'may'
duration = 1042
campaign = 1
pdays = -1
previous = 0
poutcome = 'unknown'

data = [[age, job, marital, education, default, balance,housing ,loan ,contact , day, month, duration, campaign, pdays, previous, poutcome]]



preds = loaded_model.predict(pd.DataFrame(data))[0]

print(preds)