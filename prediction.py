import joblib

def predict(data):
    model = joblib.load('gwo_decision_tree.sav')
    return model.predict(data)
