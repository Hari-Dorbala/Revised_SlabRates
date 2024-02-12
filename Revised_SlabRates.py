#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Function to load and concatenate files
def load_and_concatenate(files):
    appended_df = pd.concat(files, ignore_index=True)
    return appended_df

# Function to create WLS model
def create_wls_model(X, y, weights):
    X = sm.add_constant(X)
    wls_model = sm.WLS(y, X, weights=weights).fit()
    return wls_model

# Function to calculate predicted price
def calculate_predicted_price(model, user_distance):
    X_pred = sm.add_constant(pd.Series(user_distance))
    y_pred = model.predict(X_pred)
    return y_pred[0]

# Function to calculate slab rates
def calculate_slab_rates(appended_df, slab_start_1, slab_end_1, slab_start_2, slab_end_2, slab_start_3, slab_end_3):
    vp_sr1 = appended_df[(appended_df['distance'] <= slab_end_1)]['pred price per km'].mean()
    vp_sr2 = appended_df[(appended_df['distance'] > slab_end_1) & (appended_df['distance'] <= slab_end_2)]['pred price per km'].mean()
    vp_sr3 = appended_df[(appended_df['distance'] > slab_end_2) & (appended_df['distance'] <= slab_end_3)]['pred price per km'].mean()
    

    sr1_rate_input = st.sidebar.number_input("Enter Slab Rate for 1st Slab:", min_value=0.0, step=1.0, value=500.0)
    

    return sr1_rate_input, vp_sr2, vp_sr3
# Streamlit app
def main():
    st.title("Slabs Based Price Estimation Application")

    st.sidebar.markdown("## Upload Files")
    uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    st.sidebar.markdown("**Note:** Please upload files pertaining to the same vehicle type.")

    if uploaded_files:
        appended_df = load_and_concatenate([pd.read_csv(file) for file in uploaded_files])

        X_column = [col for col in appended_df.columns if 'distance' in col.lower()][0]
        X = appended_df[X_column]
        y = appended_df['price']
        weights = np.log(appended_df[X_column])

        wls_model = create_wls_model(X, y, weights)

        # Calculate predicted prices and slab rates
        X_pred = sm.add_constant(X)
        y_pred = wls_model.predict(X_pred)
        appended_df['Predicted Price'] = y_pred
        appended_df['difference'] = appended_df['Predicted Price'] - appended_df['price']
        appended_df['pred price per km'] = appended_df['Predicted Price'] / appended_df[X_column]

        # Allow the user to input slab distances for each slab
        slab_start_1 = 0.0
        slab_end_1 = st.sidebar.number_input("Enter End Distance for 1st Slab:", min_value=0.0, step=1.0, value=5.0)
        slab_start_2 = slab_end_1
        slab_end_2 = st.sidebar.number_input("Enter End Distance for 2nd Slab:", min_value=slab_start_2, step=1.0, value=15.0)
        slab_start_3 = slab_end_2
        slab_end_3 = st.sidebar.number_input("Enter End Distance for 3rd Slab:", min_value=slab_start_3, step=1.0, value=75.0)

        sr1_rate_input, vp_sr2, vp_sr3 = calculate_slab_rates(appended_df, slab_start_1, slab_end_1, slab_start_2, slab_end_2, slab_start_3, slab_end_3)

        # Display slab rates in the left sidebar
        st.sidebar.markdown("## Slab Rates:")
        st.sidebar.write(f"Slab Rate 1: ₹{int(sr1_rate_input):,}")
        st.sidebar.write(f"Slab Rate 2: ₹{int(vp_sr2):,}")
        st.sidebar.write(f"Slab Rate 3: ₹{int(vp_sr3):,}")

        st.markdown("### Estimated Price for User's Distance:")
        user_distance = st.number_input("Enter the distance for price estimation:", min_value=0.0, step=1.0)

        # Calculate user price based on slab rates
        user_price = 0

        if user_distance <= slab_end_1:
            user_price = sr1_rate_input
        elif slab_end_1 < user_distance <= slab_end_2:
            user_price = sr1_rate_input + (user_distance - slab_end_1) * vp_sr2
        else: 
            user_price =  (user_distance ) * vp_sr3
        

        appended_df['Calculated Price'] = 0
        appended_df.loc[appended_df['distance'] <= slab_end_1, 'Calculated Price'] = sr1_rate_input
        appended_df.loc[(slab_end_1 < appended_df['distance']) & (appended_df['distance'] <= slab_end_2), 'Calculated Price'] = sr1_rate_input + (appended_df['distance'] - slab_end_1) * vp_sr2
        appended_df.loc[(slab_end_2 < appended_df['distance']) & (appended_df['distance'] <= slab_end_3), 'Calculated Price'] = (appended_df['distance']) * vp_sr3
        
        

            

        st.write(f"The Estimated price for a distance of {user_distance} is: ₹{int(user_price):,}")

        # Graph of predicted vs actual values
        st.subheader("Graph of Predicted vs Actual Values")
        st.scatter_chart(appended_df[['Predicted Price', 'price']].head(100))

        # Table of distance, actual price, predicted price, and calculated price
        st.subheader("Table of Prices")
        prices_table = appended_df[['distance', 'price', 'Predicted Price', 'Calculated Price']].head(100)
        st.table(prices_table)

if __name__ == "__main__":
    main()


