# app.py
import os 
import json 
import joblib
import numpy as no
model path = os.getenv("MODEL_PATH", "model/iris_model.pkl") #adjust filename if needed 

#-----App----
app = flask(__name__)

#load once at startup 
try:
    model = joblib.load(MODEL_PATH)
except exception as e :
    #fail fast with a helpful message 
    raise RunetimeError(f"could not load model from{MODEL_PATH}:{E}")

@app.get("/health")
def health():
    return{"status": "ok"}, 200


@app.post("/predict")
def predict():
    """
    Accepts either:
    {"input": [[...feature vector...], [...]]} 
    or
    {"input": [...feature vector...]}
    """
    try:
        payload = request.get_json(force=True)
        x = payload.get("input")
        if x is None:
            return {"error": "missing 'input' in request"}, 400
        
        #normalize to 2d array
        if isinstance(x, list) and (len(x) > 0) and isinstance(x[0], list):
            x = [x]
        
        x = np.array(x, dtype=float)
        pred = model.predict(x)
        #if your model returns numpy types, convert to python
        pred = pred.tolist()
        return {"predictions": pred}, 200
    
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    #local dev only; Render will run with gunicorn (see startcommand below)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))