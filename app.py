# Import streamlit and pandas
import streamlit as st
import pandas as pd
from model import *

# Create a title for the app
st.title("Data Editor App")

# Create a text input for the user to enter the table name
table_name = st.text_input("Enter the data set name:")

wave_length = st.selectbox("Select data wave lengths:",options=["SWIR & LWIR", "SWIR", "LWIR"])

# Creating a list of swir and lwir columns
swir_columns = [f'SWIR{i}' for i in range(1, 21)]
lwir_columns = [f'LWIR{i}' for i in range(1, 21)]

# Combining all the columns including the 'depth' column
if wave_length == "SWIR":
    columns = ['Depth'] + swir_columns + ['Mineral'] 
elif wave_length == "LWIR":
    columns = ['Depth'] + lwir_columns + ['Mineral'] 
else:
    columns = ['Depth'] + swir_columns + lwir_columns + ['Mineral'] 

# Creating an empty DataFrame with the above columns
df = pd.DataFrame(columns=columns)

# Creating a new row with all zeros
new_row = [0] * len(columns)

# Adding the row to the DataFrame using loc
df.loc[0] = new_row
df.loc[1] = new_row
df.loc[2] = new_row

# Create a data editor for the user to enter the table data
table_data = st.data_editor(df, height=200, num_rows="dynamic")

# Load the table data into a pandas dataframe
df = pd.DataFrame(table_data)

model = st.selectbox("Select model:",options=["Linear Regression", "Random Forest", "Gradiant Boosted"])


if st.button("Run Model"):
    if model == "Linear Regression":
        predictions, elements_r2 = liner_reg_training(df)
        r2_df = pd.DataFrame(elements_r2)
        st.write(model + " R^2: " + str(r2_df[1][1]))
        predictions = predictions.drop('Depth_pred', axis=1)
        st.write(predictions)
    elif model == "Random Forest":
        st.write("Currently unsupported")
    else:
        st.write("Currently unsupported")



