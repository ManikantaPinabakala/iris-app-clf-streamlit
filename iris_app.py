# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

@st.cache()
def prediction(sepal_length, sepal_width, petal_length, petal_width):
  species = svc_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"


# Add title widget
st.title("Iris Flower Species Prediction App")  

# Add 4 sliders and store the value returned by them in 4 separate variables.
s_len = st.slider("Sepal Length", 0.0, 10.0)
s_wid = st.slider("Sepal Width", 0.0, 10.0)
p_len = st.slider("Petal Length", 0.0, 10.0)
p_wid = st.slider("Petal Width", 0.0, 10.0)

# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'. 
# Print the values of 'species_type' and 'score' variables using the 'st.text()' function.
if st.button("Predict"):
	species_type = prediction(s_len, s_wid, p_len, p_wid)
	st.write("Species predicted:", species_type)
	st.write("Accuracy score of this model is:", score)