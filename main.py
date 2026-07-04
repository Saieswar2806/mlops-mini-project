from fastapi import FastAPI
from pydantic import BaseModel
import joblib

#Load the model
model=joblib.load("iris_model.pkl")

class_names = ["Setosa", "Versicolor", "Virginica"]

#Initialize the API
app=FastAPI(title="Iris Flower Predictor API")

#Define the input data
class FlowerData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

#Create the prediction end point
@app.post("/predict")
def predict_flower(data:FlowerData):

    # Format the data into a 2D list for the model to read
    input_data=[[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    #feed the input to model to get prediction
    prediction= model.predict(input_data)

    #convert the prediction number into the flower name
    predicted_class_id = int(prediction[0])
    flower_name = class_names[predicted_class_id]

    return {
        "status": "success",
        "predicted_flower_type": flower_name
    }