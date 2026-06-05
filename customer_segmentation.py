from flask import Flask,redirect,render_template,flash,url_for,request

app = Flask(__name__)

@app.route("/",methods = ['GET','POST'])
def home():
    return render_template('customer_segmentation_home.html')

@app.route('/predict',)
def predict():
    if request.method == 'POST':
