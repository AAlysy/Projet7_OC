# -*- coding: utf-8 -*-
# TO RUN : $python api/api_flask.py
import pandas as pd
import sklearn
import joblib
from flask import Flask, jsonify, request
import json
import streamlit as st
app = Flask(__name__)

@st.cache
def get_data(filename):
    test = pd.read_csv(filename)
    return test

test = get_data("data_test.csv")
test_sample = test.sample(frac=.1, random_state=23)
test_sample = test_sample.drop("SK_ID_CURR.1", axis=1)
colonnes = (test_sample.columns)[1:]

###############################################################
app = Flask(__name__)

@app.route("/")
def loaded():
    return "API, models and data loaded…"

@app.route('/api/personal_data/')
# Test : http://127.0.0.1:5000/api/personal_data?SK_ID_CURR=346770
def personal_data():
    # Parsing the http request to get arguments (applicant ID)
    input_id = int(float(request.args.get('SK_ID_CURR')))
    # Getting the personal data for the applicant (pd.Series)
    data_client = test_sample[test_sample['SK_ID_CURR'] == input_id]
    data_client_1= data_client[colonnes]

    # Converting the pd.Series to JSON
    personal_data_json = json.loads(data_client_1.to_json(orient='records'))
    print(personal_data_json)
    
    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': personal_data_json
     })

@app.route('/api/aggregations/')
# Test : http://127.0.0.1:5000/api/aggregations
def aggregations():

    # Converting the pd.Series to JSON
    data_agg_json = json.loads(data_agg.to_json())

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': data_agg_json
     })

@app.route('/api/features_desc/')
# Test : http://127.0.0.1:5000/api/features_desc
def send_features_descriptions():

    # Converting the pd.Series to JSON
    features_desc_json = json.loads(features_desc.to_json())

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': features_desc_json
     })

@app.route('/api/features_imp/')
# Test : http://127.0.0.1:5000/api/features_imp
def send_features_importance():
    features_importance = pd.Series(surrogate_model.feature_importances_, index=data_original_le.columns)
    
    # Converting the pd.Series to JSON
    features_importance_json = json.loads(features_importance.to_json())

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': features_importance_json
     })

@app.route('/api/local_interpretation/')
# Test : http://127.0.0.1:5000/api/local_interpretation?SK_ID_CURR=346770
def send_local_interpretation():

    # Parsing the http request to get arguments (applicant ID)
    SK_ID_CURR = int(request.args.get('SK_ID_CURR'))

    # Getting the personal data for the applicant (pd.DataFrame)
    local_data = data_original_le.loc[SK_ID_CURR:SK_ID_CURR]

    # Computation of the prediction, bias and contribs from surrogate model
    prediction, bias, contribs = ti.predict(surrogate_model, local_data)
    
    # Creating the pd.Series of features_contribs
    features_contribs = pd.Series(contribs[0], index=data_original_le.columns)

    # Converting the pd.Series to JSON
    features_contribs_json = json.loads(features_contribs.to_json())

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'prediction': prediction[0][0],
        'bias': bias[0],
        'contribs': features_contribs_json,
     })



#################################################
if __name__ == "__main__":
    app.run(debug=False)