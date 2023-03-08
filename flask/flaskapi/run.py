from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import pandas as pd
import datetime

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__, template_folder='Templates')

@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    features = ['annual_income', 'birthday', 'days_employment', 'family_size',
       'work_phone', 'phone', 'email', 'car', 'property', 'education',
       'gender_M', 'gender_F', 'income_Student', 'income_Pensioner',
       'income_State servant', 'income_Commercial associate',
       'income_Working', 'housing_Municipal apartment',
       'housing_With parents', 'housing_Office apartment',
       'housing_Rented apartment', 'housing_House / apartment',
       'housing_Co-op apartment', 'occupation_Cleaning staff',
       'occupation_others', 'occupation_Security staff',
       'occupation_Medicine staff', 'occupation_Managers',
       'occupation_Sales staff', 'occupation_Cooking staff',
       'occupation_Core staff', 'occupation_Laborers',
       'occupation_Low-skill Laborers', 'occupation_Waiters/barmen staff',
       'occupation_IT staff', 'occupation_Accountants',
       'occupation_Secretaries', 'occupation_High skill tech staff',
       'occupation_Drivers', 'occupation_Realty agents',
       'occupation_Private service staff', 'marital_Single / not married',
       'marital_Married', 'marital_Civil marriage', 'marital_Separated',
       'marital_Widow']
    X = pd.DataFrame([[0]*46])
    X.columns = features

    if request.method == "POST":
        X['annual_income'] = float(request.form["annual_income"])
        X['family_size'] = float(request.form['family_size'])
        X['work_phone'] = int(request.form['work_phone'])
        X['phone'] = int(request.form['phone'])
        X['email'] = int(request.form['email'])
        X['car'] = int(request.form['car'])
        X['property'] = int(request.form['property'])
        X['education'] = int(request.form['education'])

        today = datetime.date.today()
        b_day = datetime.datetime.strptime(
                          request.form['birthday'], '%Y-%m-%d').date()
        e_day = datetime.datetime.strptime(
                          request.form['days_employment'], '%Y-%m-%d').date()
        X['birthday'] = (b_day - today).days
        if (e_day - today).days > 0:
            X['days_employment'] = 0
        else:
            X['days_employment'] = (e_day - today).days

        if request.form['gender'] == 'male':
            X['gender_M'] = 1
        elif request.form['gender'] == 'female':
            X['gender_F'] = 1

        if request.form['income'] == 'student':
            X['income_Student'] = 1
        elif request.form['income'] == 'pensioner':
            X['income_Pensioner'] = 1
        elif request.form['income'] == 'state servant':
            X['income_State servant'] = 1
        elif request.form['income'] == 'commercial associate':
            X['income_Commercial associate'] = 1
        elif request.form['income'] == 'working':
            X['income_Working'] = 1

        if request.form['housing'] == 'ma':
            X['housing_Municipal apartment'] = 1
        elif request.form['housing'] == 'p':
            X['housing_With parents'] = 1
        elif request.form['housing'] == 'oa':
            X['housing_Office apartment'] = 1
        elif request.form['housing'] == 'ra':
            X['housing_Rented apartment'] = 1
        elif request.form['housing'] == 'h/a':
            X['housing_House / apartment'] = 1
        elif request.form['housing'] == 'ca':
            X['housing_Co-op apartment'] = 1

        
        if request.form['occupation'] == 'cs':
            X['occupation_Cleaning staff'] = 1
        elif request.form['occupation'] == 'other':
            X['occupation_others'] = 1
        elif request.form['occupation'] == 'security':
            X['occupation_Security staff'] = 1
        elif request.form['occupation'] == 'med':
            X['occupation_Medicine staff'] = 1
        elif request.form['occupation'] == 'm':
            X['occupation_Managers'] = 1
        elif request.form['occupation'] == 'sales':
            X['occupation_Sales staff'] = 1
        elif request.form['occupation'] == 'cooking':
            X['occupation_Cooking staff'] = 1
        elif request.form['occupation'] == 'core':
            X['occupation_Core staff'] = 1
        elif request.form['occupation'] == 'laborer':
            X['occupation_Laborers'] = 1
        elif request.form['occupation'] == 'low':
            X['occupation_Low-skill Laborers'] = 1
        elif request.form['occupation'] == 'waiter':
            X['occupation_Waiters/barmen staff'] = 1
        elif request.form['occupation'] == 'it':
            X['occupation_IT staff'] = 1
        elif request.form['occupation'] == 'accountants':
            X['occupation_Accountants'] = 1
        elif request.form['occupation'] == 'sec':
            X['occupation_Secretaries'] = 1
        elif request.form['occupation'] == 'high':
            X['occupation_High skill tech staff'] = 1
        elif request.form['occupation'] == 'd':
            X['occupation_Drivers'] = 1
        elif request.form['occupation'] == 'realty':
            X['occupation_Realty agents'] = 1
        elif request.form['occupation'] == 'ps':
            X['occupation_Private service staff'] = 1


        if request.form['marital'] == 'single':
            X['marital_Single / not married'] = 1
        elif request.form['marital'] == 'm':
            X['marital_Married'] = 1
        elif request.form['marital'] == 'cm':
            X['marital_Civil marriage'] = 1
        elif request.form['marital'] == 'separated':
            X['marital_Separated'] = 1
        elif request.form['marital'] == 'w':
            X['marital_Widow'] = 1

        print(X.values)
        # raise Exception
        pred = model.predict(X)
        if pred == 1:
            pred = 'declined!'
        else:
            pred = 'approved!'

    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
