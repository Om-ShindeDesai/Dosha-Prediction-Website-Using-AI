import pickle
from flask import Flask, render_template, request
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
EncoderDIct = pickle.load(open('LabelEncoderDict.pkl', 'rb'))


app = Flask(__name__, template_folder='Templates', static_folder='static')
@app.route('/')
def Testing():
    return render_template('Testing.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        print("to_predict_list", to_predict_list)
        prediction = model.predict(np.array(to_predict_list).reshape(1, -1))
        print("prediction", prediction)
        if prediction == [0]:
            prediction = 'Kapha'
        elif prediction == [1]:
             prediction ='Pitta'
        else:
            prediction = 'Vatta'
        return render_template("result.html", prediction=prediction)


if __name__=='__main__':
    app.run(debug=True)