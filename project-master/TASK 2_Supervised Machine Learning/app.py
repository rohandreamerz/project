import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if (prediction>100):
        output = (1/(1+math.exp(-prediction)))*100
        remark = "You are Perfect! Keep it up!"
    else:
        output = round(prediction[0],2)
        remark = "Awesome! But You can still do better!"
    
    
   
    return render_template('index.html', prediction_text='Predicted Percentage Score is : {}'.format(output), result='{}'.format(remark))


if __name__ == "__main__":
    app.run(debug=True)
