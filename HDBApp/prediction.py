import joblib

def predict(data):
    gb = joblib.load("C:\\Users\\Krithika JK\\Documents\\GitHub\\FYP\\HDBApp\\gradient_boosting_model.joblib")
    return gb.predict(data)