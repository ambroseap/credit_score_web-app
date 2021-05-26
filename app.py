import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = [ "NAME_CONTRACT_TYPE", "CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL",
                       "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE","  NAME_EDUCATION_TYPE",
                        "NAME_HOUSING_TYPE","DAYS_BIRTH","DAYS_EMPLOYED","FLAG_MOBIL","OCCUPATION_TYPE	",
                        "ORGANIZATION_TYPE","NAME_FAMILY_STATUS" ]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 1:
        res_val = "<<<    GOOD     >>>"
    else:
        res_val = "<<<    BAD     >>> "


    return render_template('index.html', prediction_text='Risk Performance is very  {}'.format(res_val))

if __name__ == "__main__":
    app.run()
