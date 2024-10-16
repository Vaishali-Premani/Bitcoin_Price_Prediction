## IMPORTING LIBRARIES
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tensorflow.keras.models import load_model
import os
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

## LOADING PRE_TRAINED MODEL
model = load_model('model/lstm_bitcoin_model.h5')

## SETTING PAGE NAME AND ICON
st.set_page_config(page_title="Bitcoin", page_icon= ':material/currency_bitcoin:', layout='wide')

## INPUT FOR CURRENCY
crypto_currency = 'BTC'
against_currency = 'USD'

## LOAD THE DATA
start = dt.datetime(2015,1,1)
end = dt.datetime.now()
df = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end)

# # Data Preprocessing
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df)

# # Prediction days and data preparation
# prediction_days = 60

# SIDEBAR NAVIGATION
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg" width="100" />
        <h2>Bitcoin Dashboard</h2>
    </div>
    """, 
    unsafe_allow_html=True
)

option = st.sidebar.radio(
    "Go to", 
    ("Home", "About Bitcoin", "Plots")
)

# CONTENT BASED ON USER SELECTION

if option == "Home":
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column ratios for better centering
    with col2:
        st.image("./images/logo2.png", use_column_width=True)

    st.markdown("<h1 style='text-align: center; color:black;'>!! Price Prediction !!</h1>", unsafe_allow_html=True)

    ## CURRENT DATE DISPLAY
    current_date = dt.datetime.now().strftime("%d-%m-%Y")  
    st.markdown(f"<h3 style='text-align: center;'>📅 Date: {current_date}</h3>", unsafe_allow_html=True)

    ## DISPLAY THE OPEN, HIGH, LOW & CLOSE VALUES OF BITCOIN FOR THE CURRENT DATE
    today_data = df.iloc[-1]  # Get the latest row of data
    ohlc_data = {
        "Open": today_data['Open'],
        "High": today_data['High'],
        "Low": today_data['Low'],
        "Close": today_data['Close']
    }

    st.markdown(
        """
        <style>
        .ohlc-box {
            background-color: #f0f2f6;
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #333;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='ohlc-box'>Open<br>${ohlc_data['Open']:.2f}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='ohlc-box'>High<br>${ohlc_data['High']:.2f}</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='ohlc-box'>Low<br>${ohlc_data['Low']:.2f}</div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='ohlc-box'>Close<br>${ohlc_data['Close']:.2f}</div>", unsafe_allow_html=True)

    ## NEXT DAY PREDICTION LOGIC
    # Prepare input for prediction (last 60 days of closing prices)
    prediction_days = 60
    model_inputs = df['Close'][-prediction_days:].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_inputs = scaler.fit_transform(model_inputs)

    # Reshape input to match LSTM's expected shape
    real_data = np.array([scaled_inputs])
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    # Predict next day's price
    prediction = model.predict(real_data)
    predicted_price = scaler.inverse_transform(prediction)[0, 0]

    # Display the prediction result
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 20px;'>
            <h2>📈 Predicted Bitcoin Price for Tomorrow: <span style='color:green;'>${predicted_price:.2f}</span></h2>
        </div>
        """, 
        unsafe_allow_html=True
    )

    ## TABLE FOR DATA OF LAST 'N' DAYS
    num_days = st.number_input("Enter the number of days to see the data:", min_value=1, max_value=len(df), step=1)

    if st.button("Show Data"):
        selected_data = df.tail(num_days).iloc[::-1]  # Reverse order to show latest at top

        st.markdown(
            f"""
            <table class='center-table'>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Open</th>
                        <th>High</th>
                        <th>Low</th>
                        <th>Close</th>
                        <th>Adj Close</th>
                        <th>Volume</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(
                        f"<tr><td>{date.strftime('%d-%m-%Y')}</td>"
                        f"<td>${row['Open']:.2f}</td>"
                        f"<td>${row['High']:.2f}</td>"
                        f"<td>${row['Low']:.2f}</td>"
                        f"<td>${row['Close']:.2f}</td>"
                        f"<td>${row['Adj Close']:.2f}</td>"
                        f"<td>{int(row['Volume'])}</td></tr>"
                        for date, row in selected_data.iterrows()
                    )}
                </tbody>
            </table>
            """, 
            unsafe_allow_html=True
        )

        
        

elif option == "About Bitcoin":
    st.title("About Bitcoin")
    st.write("""
    Cryptocurrency is a digital or virtual form of money that uses cryptography
    for security. Unlike traditional currencies, it operates independently of a
    central bank. Cryptocurrencies leverage blockchain technology, a decentralized
    ledger that ensures transparency and security by recording transactions across a
    network of computers.
    The first cryptocurrency was Bitcoin, founded in 2009 and remains the best
    known even today. Bitcoin had a market valuation of approximately
    $ 68,873.31 (BTC / USD) as of July 2024, accounting for approximately 55.5% of 
    the cryptocurrency market.
    Bitcoin is a decentralized digital currency that enables peer-to-peer
    transactions without the need for intermediaries like banks or governments. 
    Introduced in 2009 by an anonymous individual or group known as Satoshi Nakamoto, 
    Bitcoin operates on a blockchain—a public ledger that records all transactions 
    securely and transparently. Bitcoin is powered by a network of miners who validate
    transactions and secure the network through a process called proof-of-work. Known 
    for its volatility, Bitcoin has evolved from being an experimental digital currency
    to a widely recognized asset class, used for payments, investments, and as a hedge 
    against inflation. With a capped supply of 21 million coins, Bitcoin is often referred
    to as "digital gold," symbolizing its role as a store of value in the evolving financial ecosystem. 

    """)

# elif option == "Plots":
#     st.title("Bitcoin Price Plots")

#     # MULTISELECT OPTION FOR CHOOSING THE FIELDS TO PLOT
#     selected_fields = st.multiselect(
#         "Select data to plot", 
#         ["Open", "Close", "High", "Low", "Volume"], 
#         default=["Close"]
#     )

#     if selected_fields:
#         st.subheader(f"Plot of {', '.join(selected_fields)} vs Time")
        
#         # PLOT USING PLOTLY EXPRESS
#         fig = px.line(df, x=df.index, y=selected_fields, 
#                       title=f"{crypto_currency} {', '.join(selected_fields)} Prices Over Time")
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.warning("Please select at least one field to plot.")
        
        
elif option == "Plots":
    st.title("Bitcoin Price Plots")

    # Define the number of days used for prediction
    prediction_days = 60  # Set this to match your model’s input window size

    # Actual vs. Predicted Plot
    if st.checkbox("Show Actual vs Predicted Prices"):
        # USER INPUT: Number of Previous Days for Comparison
        num_days_for_prediction = st.number_input(
            "Enter the number of days for actual vs predicted comparison:",
            min_value=prediction_days, max_value=len(df), step=1, value=100
        )

        # Display loading spinner while generating the plot
        with st.spinner("Generating Actual vs Predicted Plot... Please wait!"):
            # Prepare data for prediction (last 'num_days_for_prediction' days)
            data_to_predict = df['Close'][-num_days_for_prediction:].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data_to_predict)

            # Generate predictions using a sliding window approach
            predicted_prices = []
            for i in range(prediction_days, len(scaled_data)):
                input_data = scaled_data[i - prediction_days:i].reshape(1, prediction_days, 1)
                predicted_price = model.predict(input_data)
                predicted_prices.append(predicted_price[0, 0])

            # Inverse transform the predicted values to original scale
            predicted_prices = scaler.inverse_transform(
                np.array(predicted_prices).reshape(-1, 1)
            ).flatten()

            # Extract actual prices corresponding to the predicted days
            actual_prices = data_to_predict[prediction_days:].flatten()

            # Create a DataFrame for plotting
            comparison_df = pd.DataFrame({
                "Date": df.index[-len(actual_prices):],
                "Actual Price": actual_prices,
                "Predicted Price": predicted_prices
            })

            # Plot actual vs predicted prices using Plotly Express
            fig = px.line(
                comparison_df, x="Date", y=["Actual Price", "Predicted Price"], 
                title="Actual vs Predicted Bitcoin Prices", 
                labels={"value": "Price (USD)", "variable": "Legend"}
            )
            st.plotly_chart(fig, use_container_width=True)

    # MULTISELECT OPTION FOR CHOOSING THE FIELDS TO PLOT
    selected_fields = st.multiselect(
        "Select data to plot", 
        ["Open", "Close", "High", "Low", "Volume"], 
        default=["Close"]
    )

    # Regular plots based on selected fields
    if selected_fields:
        st.subheader(f"Plot of {', '.join(selected_fields)} vs Time")
        fig = px.line(df, x=df.index, y=selected_fields, 
                      title=f"{crypto_currency} {', '.join(selected_fields)} Prices Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one field to plot.")
