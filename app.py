from flask import Flask,request,render_template,session,redirect,flash
import pickle
import numpy as np
import pandas as pd
# import joblib
app = Flask(__name__)
app.secret_key="supersecret"

df_full = pd.read_csv('C:/creditcard.csv')
df_full.columns = df_full.columns.str.strip()

#Load model
with open("model.pkl","rb") as f:
    best_model = pickle.load(f)


with open("all_features.pkl","rb") as f:
    all_features = pickle.load(f)

with open("scaler.pkl",'rb') as f:
    scaler = pickle.load(f)

# from tensorflow.keras.models import load_models
# autoencoder = load_model("autoencoder.h5")

autoencoder = pickle.load(open("autoencoder.pkl","rb"))
v_means = pickle.load(open("v_means.pkl","rb"))
try:
    with open ("best_threshold.pkl",'rb') as f:
        best_threshold = pickle.load(f)
except:
    best_threshold = 0.7

v_features = [f"V{i}" for i in range(1,29)]
def create_features(amount, time, is_night, is_high_amount):

    df = pd.DataFrame()

    # ===== BASIC INPUT =====
    df["Amount"] = [amount]
    df["Time"] = [time]

    # ===== AUTO FILL V FEATURES (IMPORTANT) =====
    for v in v_features:
        df[v] = v_means[v]   # since user doesn't input these

    # ===== DERIVED =====
    df["Time_hour"] = time / 3600
    df["Amount_log"] = np.log1p(amount)
    df["Amount_hour_interaction"] = amount * df["Time_hour"]
    df["is_night"] = is_night

    # ===== V BASED FEATURES =====
    df["V_mean"] = df[v_features].mean(axis=1)
    df["V_std"] = df[v_features].std(axis=1)
    df["V_max"] = df[v_features].max(axis=1)
    df["V_min"] = df[v_features].min(axis=1)
    df["V_range"] = df["V_max"] - df["V_min"]
    df["V_sum"] = df[v_features].sum(axis=1)
    df["V_abs_sum"] = df[v_features].abs().sum(axis=1)

    df["V_outlier_count"] = (df[v_features].abs() > 2).sum(axis=1)
    df["V_positive_count"] = (df[v_features] > 0).sum(axis=1)
    df["V_negative_count"] = (df[v_features] < 0).sum(axis=1)
    df["V_pos_neg_ratio"] = df["V_positive_count"] / (df["V_negative_count"] + 1)

    # ===== AMOUNT FEATURES =====
    mean_amt = df_full["Amount"].mean()
    std_amt = df_full["Amount"].std()

    df["Amount_zscore"] = (amount - mean_amt) / std_amt
    df["Amount_to_mean"] = amount / mean_amt
    df["Amount_diff"] = amount - mean_amt
    
    df["is_high_amount"] = is_high_amount

    return df


# ================= AUTOENCODER =================
def get_recon_error(df):
    ae_features = v_features + ["Amount", "Time"]

    scaled = scaler.transform(df[ae_features])
    pred = autoencoder.predict(scaled)

    error = np.mean((scaled - pred) ** 2, axis=1)
    return error


@app.route('/')
def welcome():
    return render_template('welcome.html',feature=all_features)

@app.route("/homepage")
def homepage():
    if 'user' not in session:
        return redirect('/login')
    return render_template("homepage.html",user = session['user'])

@app.route("/login",methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        # is_guest = session.get('guest',False)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        #check user already exist or not
        cursor.execute(
                "SELECT * FROM customers WHERE email= %s", (email,))

        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user:
            if user[3] == password :
                session['user'] = email
                session.pop('guest',None)
                return redirect('/homepage')
            else:
                flash('Invalid email or password !')
                return redirect('/login')
        else:
            flash('User does not exist ! Please sign up ')
            return redirect("/signup")

    return render_template("login.html")

@app.route("/signup",methods=["GET","POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        mobile = request.form["mobile"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()
        
        # check user already exist or not
        cursor.execute("SELECT * FROM customers WHERE email = %s",(email,))
        user_existing = cursor.fetchone()
        
        if user_existing:
            flash('User already exist! Please login.')
            cursor.close()
            conn.close()
            return redirect('/login')
        else:
            #Data insert into Mysql
            cursor.execute("INSERT INTO customers (email,mobile,password) VALUES (%s, %s, %s)",
                        (email,mobile,password))
            conn.commit()
            cursor.close()
            conn.close()
            flash('Sign up success !')
            return redirect('/login')
        
        
        # return render_template('login.html')
    return render_template("signup.html")


@app.route("/predict",methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')
    
    
    is_guest = session.get('guest',False)
    try:
        amount = float(request.form.get('amount',0))
        hour = float(request.form.get('hour',0))
        is_night = int(request.form.get('is_night',0))
        is_high_amount = float(request.form.get('is_high_amount',0))

        df = create_features(amount,hour,is_night,is_high_amount)

        df['recon_error'] = get_recon_error(df)

        for col in all_features:
            if col not in df.columns:
                df[col] = 0

        df = df[all_features]
        # ================= PREDICTION =================
        prob = best_model.predict_proba(df)[0][1]


        reason = []

        if prob > best_threshold:
            reason.append("Unusual transaction pattern detected")
        if amount > 2000 or amount <= 20000:
            reason.append("High transaction amount")
            prob += 0.1
        if hour < 6 or hour > 22:
            reason.append("Unusual transaction time ")

        reason_text = " ,".join(reason) if reason else 'Normal transaction behavior'

        # 🚨 HIGH AMOUNT RULES FIRST (Priority)
        if amount >= 1000000 :
            reason.append("Very high amount,verify user or freeze account")
            risk = "Critical Risk,verify from user"
            prob = max(prob, 0.9)

        elif amount > 100000:
            reason.append("High transaction amount")
            risk = "Very High Risk,verify from user"
            prob = max(prob, 0.7)

# 📊 NORMAL ML-BASED RISK (ONLY if above not triggered)
        else:
            if prob >= 0.7:
                risk = 'High Risk'
            elif prob >= 0.4:
                risk = 'Moderate Risk'
            elif prob >= 0.2:
                risk = 'Low Risk'
            else:
                risk = 'Very Low Risk'
        prediction = 1 if prob >= best_threshold else 0



        #Result
        result = "Fraud Transaction" if prediction == 1 else "Normal Transaction"

        if not is_guest:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("INSERT INTO predictions (email,amount,result) VALUES (%s,%s,%s)",(session['user'],float(amount),str(result)))

            conn.commit()
            cursor.close()
            conn.close()

        return render_template("predict.html",prediction_text = f'{result} | Risk Level: {risk}  | reason = {reason_text}',score = round(prob * 100,2) if prob else 0)
    

    except Exception as e:
        return render_template("predict.html",prediction_text = f"Error : {e}", score = None,reason = None)
    
@app.route("/about")
def about():
    if 'user' not in session:
        return redirect('/login')
    return render_template("about.html")

@app.route("/contact")
def contact():
    if 'user' not in session:
        return redirect('/login')
    return render_template("contact.html")

@app.route("/dashboard")
def dashboard():
    is_guest = session.get('guest',False)
    if 'user' not in session:
        return redirect('/login')
    if  is_guest == True :
        flash('Guest user can not access Dashboard ')
        return redirect('/login')
    else:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM predictions WHERE email= %s",(session['user'],))
        data =cursor.fetchall()

        cursor.close()
        conn.close()
        return render_template("dashboard.html",data = data,user = session['user'])


@app.route("/predict-page")
def predict_page():
    if 'user' not in session:
        return redirect('/login')
    return render_template("predict.html")


@app.route("/logout")
def logout():
    session.pop('user',None)
    return render_template('login.html')

@app.route('/guest')
def guest_mode():
    # is_guest = session.get('guest',False)
    session['user'] = 'guest'
    session['guest'] = True

    return redirect('/predict-page')

import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Vikas@250342",
        database="fraud_db"
    )



if __name__ == "__main__":
    app.run(debug=True)