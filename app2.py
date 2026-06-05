from flask import Flask,render_template,request,session,url_for,Response,redirect

app = Flask(__name__)
app.secret_key = "supersecret"

@app.route("/",methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username ==  "admin" and password == "123":
            session["user"] == username
            return redirect(url_for("Welcome"))
        else:
            return Response("In-valid credentials,try again",mimetype="text/plain")
    # return "Welcome to my webpage"
    return render_template("level3.html")
@app.route("/about")
def about():
    return "This is about page"

@app.route("/products")
def products():
    return "This is our products"

@app.route("/contact us")
def contact():
    return "This is our contact page"

if __name__ == "__main__":
    app.run(debug=True)