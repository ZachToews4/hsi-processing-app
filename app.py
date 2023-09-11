# Import streamlit and pandas
import streamlit as st
st.set_page_config(layout="wide")

#import plotly.express as px
import pandas as pd
import plotly.express as px
from model import *

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5streamlit runrem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

# Create a title for the app
st.markdown("<h1 style='text-align: center;'>HSI Processing App</h1>", unsafe_allow_html=True)

#create layout columns
col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
col4, col5 = st.columns([0.5,0.5])

# Create a text input for the user to enter the table name
row_num = col2.number_input("Number of rows:", value=350)
col_num_hsi = col4.number_input("Number of HSI columns:", value=40)
col_num_train = col5.number_input("Number of property columns (e.g. XRF, XRD, Porosity, Hardnsee, etc.):", value=1, disabled=True)

# create columns for the data frame
columns_hsi = ["Depth"]
for i in range(col_num_hsi):
    columns_hsi.append(f"hsi_{i}")

columns_train = ["Depth"]
for i in range(col_num_train):
    columns_train.append(f"train_{i}")
    
# Creating an empty DataFrame with the above columns
df_hsi = pd.DataFrame(columns=columns_hsi)
df_train = pd.DataFrame(columns=columns_train)

# Creating a new row with all zeros
new_row_hsi = [0] * len(columns_hsi)
new_row_train = [0] * len(columns_train)


# Adding the row to the DataFrame using loc
i=0
while i < row_num:
    df_hsi.loc[i] = new_row_hsi
    i = i + 1

i=0
while i < row_num:
    df_train.loc[i] = new_row_train
    i = i + 1

# Create a data editor for the user to enter the table data
table_data_hsi = col4.data_editor(df_hsi.applymap(lambda x: f"{x:.1f}"), height=200, num_rows="dynamic", hide_index=True)
table_data_train = col5.data_editor(df_train.applymap(lambda x: f"{x:.1f}"), height=200, num_rows="dynamic", hide_index=True)

# Load the table data into a pandas dataframe
table_data_hsi = table_data_hsi.reset_index(drop=True)
df_hsi = pd.DataFrame(table_data_hsi)
table_data_train = table_data_train.reset_index(drop=True)
df_train = pd.DataFrame(table_data_train)

model = col4.selectbox("Select model:",options=["Linear Regression", "Random Forest", "Gradient Boosting"])

if model == "Random Forest":
    bootstrap = col4.selectbox("Bootstrap", ["True", "False"], index=0)
if model == "Gradient Boosting":
    learning_rate = col4.number_input("Learning Rate (float)", value=0.1)
if model == "Random Forest" or model == "Gradient Boosting":
    max_depth = col4.text_input("Max Depth (int or None)", value=80)
    if max_depth == "None":
        max_depth = None
    else:
        max_depth = int(max_depth)
    max_features = col4.text_input("Max Features ({“sqrt”, “log2”, None}, int or float)", value='sqrt')
    if max_features != "None" and max_features != "sqrt" and max_features != "log2":
        max_features = float(max_features)
        if max_features > 1:
            max_features = int(max_features)
    elif max_features == "None":
        max_features = None
    min_samples_leaf = col4.number_input("Min Samples Leaf (int/min number or float/min fraction)", value=1)
    min_samples_split = col4.number_input("Min Samples Split (int/min number or float/min fraction)", value=10)
    n_estimators = col4.number_input("N Estimators (int)", value=500)

test_data = pd.DataFrame()
if col4.button("Run Model"):
    if model == "Linear Regression":
        test_data, r2_values = linear_regression(df_hsi, df_train)
    elif model == "Random Forest":
        test_data, r2_values = random_forest_regression(df_hsi, df_train, bootstrap=bootstrap, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)
    elif model == "Gradient Boosting":
        test_data, r2_values = gradient_boosting_regression(df_hsi, df_train, learning_rate=learning_rate, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)
    st.write(test_data)
    st.write(r2_values[0])

    fig = px.scatter(test_data, x="true", y="predicted", trendline="ols")
    fig.update_layout(
        title="True vs Predicted",
        xaxis_title="True",
        yaxis_title="Predicted",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    st.plotly_chart(fig)

# Create a button to save the table data to a csv file
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

csv = convert_df(test_data)

table_name = col4.text_input("Enter the data set name:", value="new_table")

st.download_button(
   "Press to Download",
   csv,
   f"{table_name}.csv",
   "text/csv",
   key='download-csv'
)





