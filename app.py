import flask
from flask import request
import sklearn
import pickle
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#app = flask.Flask(__name__)
#app.config["DEBUG"] = True
app = Flask(__name__, template_folder='template')
lightgb = pickle.load(open('lgb.pkl', 'rb'))

from flask_cors import CORS
CORS(app)

# main index page route
@app.route('/')
def default():
    return render_template("homepage.html")



@app.route('/predict')

def get_data():
    internationalplan = request.form.get('international plan')
    voicemailplan = request.form.get('voice mail plan')
    numbervmailmessages = request.form.get('number vmail messages')
    totalintlcalls = request.form.get('total intl calls')
    customerservicecalls= request.form.get('customer service calls')
    
    d_dict = {'total intl calls': ['total intl calls'], 'number vmail messages': ['number vmail messages'], 'Internationalplan_yes': [0],
              'Internationalplan_no': [0], 'Voicemailplan_yes': [0], 'Voicemailplan_no': [0], 'customer_call_0': [0],
              'customer_call_1': [0], 'customer_call_2': [0], 'customer_call_3': [0], 'customer_call_4': [0],
              'customer_call_5': [0], 'customer_call_6': [0], 'customer_call_7': [0],
              'customer_call_8': [0], 'customer_call_9': [0]}
    replace_list = [internationalplan,voicemailplan,customerservicecalls]

    for key, value in d_dict.items():
        if key in replace_list:
            d_dict[key] = 1


    return pd.DataFrame.from_dict(d_dict, orient='columns')

def feature_imp(model, data):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_5 = indices[:5]
    data = data.iloc[:, top_5]
    return data

def min_max_scale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler.fit(data)
    data_scaled = scaler.fit_transform(data.values.reshape(5, -1))
    data = data_scaled.reshape(-1, 5)
    return pd.DataFrame(data)

@app.route('/send', methods=['POST'])
def show_data():
    df = get_data()
    featured_data = feature_imp(lightgb, df)
    scaled_data = min_max_scale(featured_data)
    prediction = lightgb.predict(scaled_data)
    outcome = 'Churner'
    if prediction == 0:
        outcome = 'Non-Churner'

    return render_template('results.html', tables = [df.to_html(classes='data', header=True)],
                           result = outcome)



if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080)