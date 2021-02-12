from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.linear_model import LinearRegression
import pickle

# Create your views here.

def HouseModelTraining(request):
    context={}
    data = pd.read_csv("House_data_preprocessed.csv")
    context["samples"] = data.shape[0]

    if request.method == 'GET':
        context["score"] = "-"

    if request.method == 'POST':
        Y = data["price"]
        X = data.drop("price", axis="columns")
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(X, Y, test_size=0.2)
        house_model = LinearRegression()
        house_model.fit(x_train, y_train)
        score = house_model.score(x_test, y_test)
        context["score"] = score
        with open( 'house_model.pickle','wb') as f:
            pickle.dump(house_model,f)

    return render(request, 'housepriceprediction/HouseModelTraining.html',context)
def HouseModelPrediction(request):
    context={}
    return render(request, 'housepriceprediction/HouseModelPrediction.html',context)