from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/literature")
def literature():
    return render_template("literature.html")


@app.route("/material")
def material():
    return render_template("material.html")


@app.route("/algorithms")
def performance():
    return render_template("algorithms.html")


@app.route("/test")
def test():
    return render_template("test.html")


@app.route("/test", methods=["POST"])
def prd():
    music = dict()
    music["danceability"] = request.form["danceability"]
    music["energy"] = request.form["energy"]
    music["loudness"] = request.form["loudness"]
    music["speechiness"] = request.form["speechiness"]
    music["acousticness"] = request.form["acousticness"]
    music["instrumentalness"] = request.form["instrumentalness"]
    music["liveness"] = request.form["liveness"]
    music["valence"] = request.form["valence"]
    music["tempo"] = request.form["tempo"]
    music["duration_ms"] = request.form["duration_ms"]
    music["key"] = request.form["key"]
    music["time_signature"] = request.form["time_signature"]
    music["mode"] = request.form["mode"]

    Input = pd.DataFrame(
        data=[[music["danceability"], music["energy"], music["loudness"], music["speechiness"], music["acousticness"],
               music["instrumentalness"], music["liveness"],
               music["valence"], music["tempo"], music["duration_ms"], music["key"], music["time_signature"],
               music["mode"]]],
        columns=['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                 'valence', 'tempo', 'duration_ms', 'key', 'time_signature', 'mode'])
    Input['key_0.0'] = 0
    Input['key_1.0'] = 0
    Input['key_2.0'] = 0
    Input['key_3.0'] = 0
    Input['key_4.0'] = 0
    Input['key_5.0'] = 0
    Input['key_6.0'] = 0
    Input['key_7.0'] = 0
    Input['key_8.0'] = 0
    Input['key_9.0'] = 0
    Input['key_10.0'] = 0
    Input['key_11.0'] = 0
    val = Input['key'][0]
    st = 'key_' + str(val) + '.0'
    Input[st] = 1
    Input.drop('key', axis=1, inplace=True)
    Input['time_signature_1.0'] = 0
    Input['time_signature_3.0'] = 0
    Input['time_signature_4.0'] = 0
    Input['time_signature_5.0'] = 0
    val = Input['time_signature'][0]
    st = 'time_signature_' + str(val) + '.0'
    Input[st] = 1
    Input.drop('time_signature', axis=1, inplace=True)
    Input['mode_0.0'] = 0
    Input['mode_1.0'] = 0
    val = Input['mode'][0]
    st = 'mode_' + str(val) + '.0'
    Input[st] = 1
    Input.drop('mode', axis=1, inplace=True)
    output = Input.shape

    Ans = predict(Input)[0]

    return render_template('test.html', output=Ans)


def predict(Input):
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    Input[
        ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
         'tempo', 'duration_ms']] = scaler.fit_transform(Input[['danceability', 'energy', 'loudness', 'speechiness',
                                                                'acousticness', 'instrumentalness', 'liveness',
                                                                'valence', 'tempo', 'duration_ms']])

    Prediction = model.predict(Input)

    return (Prediction)


if __name__ == "__main__":
    app.run(debug=True)
