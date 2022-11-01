import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from lime import lime_tabular
import requests
import json
from lime import lime_text

@st.cache
def get_data(filename):
    test = pd.read_csv(filename)
    return test

test = get_data("data_test.csv")
test_sample = test.sample(frac=.1, random_state=23)
test_sample = test_sample.drop("SK_ID_CURR.1", axis=1)
colonnes = (test_sample.columns)[1:]
chosen_feats = (test_sample.columns)[1:]

st.set_option('deprecation.showPyplotGlobalUse', False)

API_URL = "http://127.0.0.1:5000/api/"

image = Image.open('image2.jpg')
pickle_in= open("prevision_credit.pkl", "rb")
load_model= pickle.load(pickle_in)
pick = open("interpretability_list.pkl", "rb")
load_exp = pickle.load(pick)

@st.cache
def get_data(filename):
    test = pd.read_csv(filename)
    return test


header = st.container()
dataset = st.container()
features= st.sidebar
model_training = st.container()
viz = st.container()
domaine = st.container()
features_importance= st.container()
feature_locale = st.container()

with header:
    st.title('Home risk credit default')


def get_personal_data(select_sk_id):
    # URL of the scoring API (ex: SK_ID_CURR = 100005)
    PERSONAL_DATA_API_URL = API_URL + "personal_data/?SK_ID_CURR=" + str(select_sk_id)

    # save the response to API request
    response = requests.get(PERSONAL_DATA_API_URL)

    # convert from JSON format to Python dict
    jsonData = json.loads(response.content.decode('utf-8'))
    content = pd.DataFrame.from_dict(jsonData)
    content = json.loads(response.content.decode('utf-8'))

    # convert data to pd.Series
    personal_data = pd.DataFrame(content['data'])
    st.write(personal_data)
    return personal_data


with model_training:

    input_id = st.number_input("Veuillez saisir l'ID du client", value=413172.)

    st.write("Exemple d'ID de bons payeurs: 229984, 227621, 347854")
    st.write("Exemple d'ID de mauvais payeurs : 285870, 150111, 188739 ")
    if input_id ==100001.:

        st.write("Entrez un ID valide")

    else :
        data_ = get_personal_data(input_id)
        data_client = data_[chosen_feats]
        st.write(data_client)
        prediction=load_model.predict(data_client)

        with features:

            data_client = get_personal_data(input_id)
            data_client = data_[chosen_feats]

            colonnes = data_client.columns

            parametre = st.selectbox("Choisissez le paramètre que vous souhaitez modifier", colonnes)
            if parametre in colonnes:
                old_value = data_client[parametre].values[0]
                new_value = st.slider(parametre, 0., 1.)
                data_client[parametre].replace(old_value, new_value, inplace=True)
            st.write(data_client)
            prediction = load_model.predict(data_client)

            if prediction == 1:
                log_reg_pred = load_model.predict_proba(data_client)[:, 0]
                st.write("Nouveau Score:", round(log_reg_pred.min() * 100, 2), "%")
            else:

                log_reg_pred = load_model.predict_proba(data_client)[:, 0]
                st.write("Nouveau score:", round(log_reg_pred.min() * 100, 2), "%")

        if prediction ==1:
            log_reg_pred = load_model.predict_proba(data_client)[:, 0]
            st.write("Le prêt n'est pas accordé, Probabilité de rembourser:", round(log_reg_pred.min() * 100, 2), "%")
        else :

            log_reg_pred = load_model.predict_proba(data_client)[:, 0]
            st.write("## Félicitations, le prêt est accordé, Probabilité de rembourser sa dette:", round(log_reg_pred.min()*100, 2),"%")





with domaine:
    data_test= test_sample.copy()
    data_test=data_test.reset_index()
    data_test_1 = data_test[colonnes]
    predict_tot= load_model.predict(data_test_1)
    predict_tot = pd.DataFrame(predict_tot, columns=["prediction"])
    data_test_1=pd.concat([data_test,predict_tot], axis=1)

    df1= data_test_1[data_test_1["prediction"] == 1]

    df1_row=df1.mean()
    Mean_non_payeur=df1_row.to_frame().T
    df0= data_test_1[data_test_1["prediction"]==0]
    df0_row = df0.mean()
    Mean_payeur = df0_row.to_frame().T


with viz :
    data_client = test_sample[test_sample['SK_ID_CURR'] == input_id]

    data_viz = pd.concat([data_client[colonnes], Mean_payeur])
    data_viz = pd.concat([data_viz, Mean_non_payeur])
    param = st.selectbox("Choisir un paramètre ", colonnes)
    data_1= data_viz[param]
    labels = ["Client", "Moyenne Payeur", "Moyenne Non Payeur"]
    sizes = [100, 200, 50]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(data_1, labels=labels, autopct="%1.1f%%")
    ax.axis("equal")

    st.pyplot(fig)

with features_importance:
    if prediction == 1:
        explainer = lime_tabular.LimeTabularExplainer(training_data= np.array(test_sample[colonnes]),
                                                  mode="classification", feature_names=colonnes)
        exp = explainer.explain_instance(data_row=data_viz[colonnes].iloc[0], predict_fn=load_model.predict_proba)
        fig1 =exp.as_pyplot_figure()
        fig2 = exp.show_in_notebook(show_table=True)
        st.pyplot(fig1)
