import pandas as pd 
import numpy as np
from kmeans import K_means
data=pd.read_csv("iris.csv")
y=data.pop("species")

kmeans=K_means()
kmeans.fit(3,data.values)
predictions =kmeans.predict(data.values)


import matplotlib.pyplot as plt
import matplotlib

for i in range(len(predictions)):
    if predictions[i]==0:
        predictions[i]="red"
    if predictions[i]==1:
        predictions[i]="blue"
    if predictions[i]==2:
        predictions[i]="green"

plt.scatter(data["sepal_length"],data["petal_length"], c=[i for i in predictions])
plt.show()
plt.scatter(data["sepal_length"],data["sepal_width"], c=[i for i in predictions])
plt.show()
plt.scatter(data["sepal_width"],data["petal_length"], c=[i for i in predictions])
plt.show()
plt.scatter(data["sepal_width"],data["petal_width"], c=[i for i in predictions])
plt.show()
plt.scatter(data["sepal_length"],data["petal_width"], c=[i for i in predictions])
plt.show()
plt.scatter(data["petal_width"],data["petal_width"], c=[i for i in predictions])
plt.show()
