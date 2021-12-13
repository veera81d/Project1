import pandas as pd
from flask import Flask, render_template, request
import joblib
app = Flask(__name__)
#model = load(model_jl)

with open("model_jl", "rb") as model:
    model_lr = joblib.load(model)

@app.route("/")
def home():
    return render_template("home.html")

input_d = {}
@app.route("/", methods =['POST'])
def inp():
    #Testing
    if request.method == 'POST':
        input_d["Bedroom"] = request.form.get('bhk')
        input_d["bath"] = request.form.get('bath')
        input_d["balcony"] = request.form.get('balcony')
        input_d["Area"] = request.form.get('area')
        input_d["Location"] = request.form.get('location')
        input_final = input_d
    df = pd.DataFrame([input_final])
    df[['Bedroom', 'bath', 'balcony']] = df[['Bedroom', 'bath', 'balcony']].apply(pd.to_numeric)
    df[['Area']] = df[['Area']].astype(float)
    dummy = pd.get_dummies(df.Location)
    dfin_final = pd.concat([df.drop(['Location'], axis=1), dummy], axis=1)
    df_input = pd.DataFrame(columns=['bath', 'balcony', 'Bedroom', 'Area', '7th Phase JP Nagar',
                                     'Bannerghatta Road', 'Electronic City', 'Electronic City Phase II',
                                     'Haralur Road', 'Hebbal', 'Hennur Road', 'Kanakpura Road',
                                     'Marathahalli', 'Raja Rajeshwari Nagar', 'Sarjapur  Road',
                                     'Thanisandra', 'Uttarahalli', 'Whitefield', 'Yelahanka'])
    for col in dfin_final:
        df_input[col] = dfin_final[col]
    df_final = df_input.fillna(0)
    price = model_lr.predict(df_final)
    return render_template("home.html", values=input_final, results=price)


if __name__ == "__main__":
    app.run(debug=True)