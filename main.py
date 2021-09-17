from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
app=Flask(__name__,static_url_path='/static')
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict',methods=['Post'])
def predict():
    int_features=[[int(x) for x in request.form.values()]]
    final_features=np.array(int_features)

    prediction=model.predict(final_features)
    output=prediction
    return render_template('index.html',prediction_text='Price ={}'.format(output))



if __name__ == '__main__':
    app.run(debug=True)
