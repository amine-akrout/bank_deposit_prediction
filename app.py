# Import Libraries
from __future__ import print_function
import os
import mlflow
from flask import Flask, render_template, request


logged_model = './catboost-model'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


app = Flask(__name__)

@app.route('/')
def entry_page():
    # Nicepage template of the webpage
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def render_message():
    try:
        # Get data input
        age = int(request.form['age'])
        job = request.form['job']
        marital = request.form['marital']
        education = request.form['education']
        default = request.form['default']
        balance = int(request.form['balance'])
        housing = request.form['housing']
        loan = request.form['loan']
        contact = request.form['contact']
        day = int(request.form['day'])
        month = request.form['month']
        duration = int(request.form['duration'])
        campaign = int(request.form['campaign'])
        pdays = int(request.form['pdays'])
        previous = int(request.form['previous'])
        poutcome = request.form['poutcome']

        data = [[age, job, marital, education, default, balance,housing ,loan ,contact , day, month, duration, campaign, pdays, previous, poutcome]]
        preds = loaded_model.predict(data)[0]
        # print(preds, file=sys.stderr)
        
        print('Python module executed successfully')
        message = 'Has the client subscribed a term deposit?   ==>  {}  !'.format(preds)
        # print(message, file=sys.stderr)
    except Exception as e:
        # Store error to pass to the web page
        message = "Error encountered. Try with other values. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(
            e.__class__, e.args, e.__doc__)
    # Return the model results to the web page
    return render_template('index.html' ,message=message)

if __name__ == '__main__':
    # app.run(debug=True , host='localhost', port=8080)
    app.run(host="0.0.0.0")
