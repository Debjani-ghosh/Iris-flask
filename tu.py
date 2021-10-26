from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/predi', methods=['POST'])
def predi():
    data1 = request.form['sepl']
    data2 = request.form['sepw']
    data3 = request.form['petl']
    data4 = request.form['petw']
    arr = np.array([[data1, data2, data3, data4]])
    data = model.predict(arr)
    return render_template('aft.html', data=data)

if __name__ == "__main__":
    app.run(debug=True)