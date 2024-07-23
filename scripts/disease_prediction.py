import pickle
import numpy as np
import pandas as pd

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def pred_result(model, X, sd, sp):
    proba = model.predict_proba(X)
    top5_idx = np.argsort(proba[0])[-5:][::-1]
    top5_proba = np.sort(proba[0])[-5:][::-1]
    top5_diseases = model.classes_[top5_idx]
    results = []
    for i in range(5):
        disease = top5_diseases[i]
        probability = top5_proba[i]
        result = {
            "Disease Name": disease,
            "Probability": probability
        }
        if disease in sd["Disease"].unique():
            disp = sd[sd['Disease'] == disease].iloc[0, 1]
            result["Disease Description"] = disp
        if disease in sp["Disease"].unique():
            c = np.where(sp['Disease'] == disease)[0][0]
            precaution_list = sp.iloc[c, 1:].dropna().tolist()
            result["Recommended Things to do at home"] = precaution_list
        results.append(result)
    return results
