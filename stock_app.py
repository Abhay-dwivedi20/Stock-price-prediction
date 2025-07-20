import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os
import joblib # For loading the scaler
import matplotlib.pyplot as plt # For plotting
import pandas_datareader.data as pdr # For fetching stock data
import pandas as pd # For date handling
from dotenv import load_dotenv # For loading API key from .env

# Load environment variables from .env file
load_dotenv()

# --- Global Configuration ---
TIME_STEP = 100 # Number of historical points for prediction input
N_FUTURE_STEPS = 30 # Number of future steps to predict
MODEL_PATH = 'my_lstm_model.h5' # As requested by user
SCALER_PATH = 'my_scaler.pkl' # As requested by user

# --- Load Model and Scaler Globally ---
# This ensures they are loaded only once when the app starts
model = None
scaler = None
tiingo_api_key_env = os.getenv("API_KEY")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded pre-trained model from '{MODEL_PATH}'")
except Exception as e:
    print(f"\n--- CRITICAL ERROR LOADING MODEL ---")
    print(f"Error loading model from '{MODEL_PATH}': {e}")
    print(f"Please ensure '{MODEL_PATH}' exists and is a valid trained Keras model.")
    print(f"The application cannot make meaningful predictions without the trained model.")
    print(f"------------------------------------\n")

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded pre-trained scaler from '{SCALER_PATH}'")
    # Basic check to ensure scaler is fitted
    if not hasattr(scaler, 'min_') or not hasattr(scaler, 'scale_'):
        raise ValueError("Loaded scaler does not appear to be fitted. It lacks 'min_' or 'scale_' attributes.")
except Exception as e:
    print(f"\n--- CRITICAL ERROR LOADING SCALER ---")
    print(f"Error loading scaler from '{SCALER_PATH}': {e}")
    print(f"Please ensure '{SCALER_PATH}' exists and is a valid, *fitted* MinMaxScaler.")
    print(f"The application cannot make meaningful predictions without the correct scaler.")
    print(f"-------------------------------------\n")

# --- Prediction Function ---
def predict_stock_price(
    ticker: str,
    optional_api_key: str, # This will be passed if the UI textbox is visible and filled
    historical_input_sequence_str: str # User-provided 100 values for prediction
) -> tuple[plt.Figure, float]:

    # Determine API Key to use
    api_key_to_use = tiingo_api_key_env
    if not api_key_to_use and optional_api_key:
        api_key_to_use = optional_api_key

    if not api_key_to_use:
        # Raise Gradio error that will be displayed in the UI
        raise gr.Error("Tiingo API Key not found. Please set TIINGO_API_KEY in your .env file or provide it in the UI.")

    if model is None or scaler is None:
        # Raise Gradio error if model/scaler failed to load at startup
        raise gr.Error("Model or scaler failed to load at startup. Check terminal for details.")

    # 1. Prepare the user-provided input sequence for prediction
    try:
        input_list = [float(x.strip()) for x in historical_input_sequence_str.split(',')]
        if len(input_list) != TIME_STEP:
            raise ValueError(f"Input sequence must contain exactly {TIME_STEP} numbers. Got {len(input_list)}.")

        # The model expects scaled input. Transform using the loaded scaler.
        # It's critical that this scaler was fitted on data with a similar range to the input.
        scaled_input_for_model = scaler.transform(np.array(input_list).reshape(-1, 1)).flatten().tolist()
        
        # Prepare for multi-step prediction
        temp_input = list(scaled_input_for_model) # Use a copy to avoid modifying original list

        lst_output_scaled = [] # To store scaled future predictions

        for i in range(N_FUTURE_STEPS):
            # Take the last TIME_STEP elements for prediction (sliding window)
            x_input = np.array(temp_input[len(temp_input) - TIME_STEP:]).reshape(1, TIME_STEP, 1)
            scaled_yhat = model.predict(x_input, verbose=0)[0][0]
            temp_input.append(scaled_yhat) # Append scalar prediction for next iteration
            lst_output_scaled.append(scaled_yhat)

        # Inverse transform predictions to original scale
        predicted_future_prices = scaler.inverse_transform(np.array(lst_output_scaled).reshape(-1, 1))
        latest_prediction = predicted_future_prices[-1][0] # Get the very last predicted value

    except ValueError as e:
        # Create a placeholder plot for error messages
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Input Sequence Error: {e}",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=14)
        ax.axis('off')
        return fig, 0.0
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Prediction Error: {e}",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=14)
        ax.axis('off')
        return fig, 0.0

    # 2. Fetch historical data for the ticker (for plotting context)
    # This is separate from the input_list used for prediction
    full_historical_data_for_plot = np.array(input_list).reshape(-1, 1) # Default to user input if fetch fails
    try:
        end_date = pd.Timestamp.now()
        # Fetch enough data to show context before the 100 input points
        # For plotting, let's try to get 200 days before the input sequence starts
        # This assumes the input_list represents the *most recent* 100 days.
        start_date_fetch = end_date - pd.DateOffset(days=TIME_STEP + N_FUTURE_STEPS + 100) # Fetch more for context
        
        df_fetched = pdr.get_data_tiingo(ticker.upper(), api_key=api_key_to_use, start=start_date_fetch, end=end_date)
        
        if not df_fetched.empty:
            full_historical_data_for_plot = df_fetched.reset_index()['close'].values.reshape(-1, 1)
            # Ensure we have at least TIME_STEP data points from fetched data to align
            if len(full_historical_data_for_plot) >= TIME_STEP:
                 # Take the last TIME_STEP data points from fetched data to align with user input for plotting
                 # This makes the plot more coherent if user input is indeed recent.
                 full_historical_data_for_plot = full_historical_data_for_plot[-TIME_STEP:]
            else:
                 print(f"Warning: Not enough fetched data ({len(full_historical_data_for_plot)} points) for '{ticker}' to match {TIME_STEP} input. Plotting only user input.")
                 full_historical_data_for_plot = np.array(input_list).reshape(-1, 1) # Fallback to just user input
        else:
            print(f"Warning: No historical data found for ticker '{ticker}' from Tiingo. Plotting only user input.")
            full_historical_data_for_plot = np.array(input_list).reshape(-1, 1) # Fallback to just user input

    except Exception as e:
        print(f"Warning: Could not fetch historical data for '{ticker}' from Tiingo: {e}. Plotting only user input.")
        # If fetching fails, we still proceed with plotting the user's provided input_list
        full_historical_data_for_plot = np.array(input_list).reshape(-1, 1)


    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(14, 8)) # Larger figure for better plot clarity

    # Plot the user's provided historical input
    # Use the length of the actual data used for plotting, which might be less than TIME_STEP if fetch failed.
    ax.plot(np.arange(len(full_historical_data_for_plot)), full_historical_data_for_plot, label='Historical Input (from UI/Tiingo)', color='blue', linewidth=2)

    # Plot predicted future data
    # Predictions start immediately after the last historical point.
    prediction_x_axis = np.arange(len(full_historical_data_for_plot), len(full_historical_data_for_plot) + N_FUTURE_STEPS)
    ax.plot(prediction_x_axis, predicted_future_prices, label=f'Predicted Next {N_FUTURE_STEPS} Steps', color='red', linestyle='--', marker='o', markersize=4)

    # Add a vertical line to indicate separation between historical and predicted
    ax.axvline(x=len(full_historical_data_for_plot) - 1, color='gray', linestyle=':', label='Last Known Data Point')

    # Customize plot
    ax.set_title(f'{ticker.upper()} Stock Price Prediction', fontsize=18, fontweight='bold')
    ax.set_xlabel('Time Steps (Relative to Input Start)', fontsize=12)
    ax.set_ylabel('Stock Price (Original Scale)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    return fig, float(latest_prediction)

# --- Gradio Interface Setup ---
# Conditional API Key input: Only show if API key not found in .env
api_key_input_component = gr.Textbox(
    label="Tiingo API Key (Optional - if not in .env)",
    type="password", # Hide input for security
    placeholder="Enter your Tiingo API key here if not set in .env",
    visible=tiingo_api_key_env is None # Only show if API key not found in .env
)

# Custom CSS for a more beautiful UI (Tailwind-like styling)
custom_css = """
body {
    font-family: 'Inter', sans-serif;
    background-color: #f3f4f6; /* Light gray background */
}
.gradio-container {
    border-radius: 1rem; /* Rounded corners for the main container */
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); /* Subtle shadow */
    overflow: hidden; /* Ensures rounded corners are applied */
}
.gr-button {
    border-radius: 0.5rem; /* Slightly rounded buttons */
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
.gr-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.gr-textbox, .gr-number {
    border-radius: 0.5rem;
    border: 1px solid #d1d5db; /* Light gray border */
    padding: 0.75rem;
}
h1, h2, h3 {
    color: #1f2937; /* Darker text for headings */
}
.gr-interface-description {
    font-size: 1rem;
    line-height: 1.5;
    color: #4b5563; /* Medium gray text */
}
.gr-interface-title {
    font-size: 2.5rem; /* Larger title */
    font-weight: 700;
    margin-bottom: 1rem;
}
.gr-column {
    padding: 1.5rem;
}
"""

# Example data for the input sequence (linear trend, sine wave, parabolic)
# These are just illustrative and should be replaced with actual stock data for better examples
example_linear = ", ".join(map(str, np.linspace(150, 160, TIME_STEP).tolist()))
example_sine = ", ".join(map(str, (150 + 10 * np.sin(np.linspace(0, 3*np.pi, TIME_STEP))).tolist()))
example_parabolic = ", ".join(map(str, (0.015 * (np.arange(TIME_STEP)**2) + 0.5 * np.arange(TIME_STEP) + 100 + np.random.normal(0, 2, TIME_STEP)).tolist()))


iface = gr.Interface(
    fn=predict_stock_price,
    inputs=[
        gr.Textbox(
            label="Stock Ticker Symbol",
            placeholder="e.g., AAPL, GOOGL, TSLA",
            value="AAPL", # Default value for convenience
            elem_id="ticker-input"
        ),
        api_key_input_component, # Conditional API key input
        gr.Textbox(
            lines=10,
            placeholder=f"Enter {TIME_STEP} comma-separated numbers representing the last {TIME_STEP} historical stock prices for prediction.",
            label=f"Last {TIME_STEP} Historical Prices (for prediction)",
            elem_id="historical-input-textbox",
            value=example_linear # Default example value
        )
    ],
    outputs=[
        gr.Plot(label=f"Stock Price Prediction (Next {N_FUTURE_STEPS} Steps)"),
        gr.Number(label="Latest Predicted Price")
    ],
    title="📈 Dynamic Stacked LSTM Stock Price Predictor",
    description=(
        "Enter a stock ticker symbol and the last 100 historical closing prices for that stock. "
        "The model will then forecast the next 30 days' prices and visualize the trend."
        "\n\n**Important Notes:**"
        f"\n1. The model (`{MODEL_PATH}`) and scaler (`{SCALER_PATH}`) are loaded from files. "
        "Ensure these files are present in the same directory as this script and were trained/fitted on data with a similar price range to your input for accurate predictions."
        "\n2. A Tiingo API key is required to fetch historical data. It's recommended to set it as an environment variable (`TIINGO_API_KEY`) in a `.env` file. "
        "If not found, an optional input box will appear for you to enter it directly."
    ),
    theme='soft',
    css=custom_css,
    allow_flagging='never',
    examples=[
        ["AAPL", "", example_linear], # Ticker, Optional API Key, Historical Input
        ["GOOGL", "", example_sine],
        ["TSLA", "", example_parabolic]
    ]
)

# --- Launch the Gradio App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    iface.launch()
    print("Gradio interface launched. Access it at the displayed URL.")
