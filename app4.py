from flask import Flask,render_template,request,session,url_for,redirect,Response
import pickle
app = Flask(__name__)
app.secret_key = "supersecret"

model = pickle("model.pkl")

@app.route("/", methods=["GET","POST"])
def login():
    if request.method =="POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # print("Debug",username ,password)

        if username == "admin" and password == "123":
            session["user"] = username
            return redirect(url_for("welcome"))
        else:
            return Response("Invalid credentials,try again",mimetype="text/plain")
    return render_template("level3.html")
        
    # return render_template("index.html")

@app.route("/welcome")
def welcome():
    if "user" in session:
        return render_template("welcome.html",user=session["user"])
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.pop("user",None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)