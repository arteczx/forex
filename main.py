import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import telegram
import time
import os

# --- Configuration ---
# MetaTrader 5 Credentials - leave blank if using passwordless login from the terminal
MT5_LOGIN = os.getenv("MT5_LOGIN", 123456)  # Replace with your account number or use environment variables
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "your_password")  # Replace with your password
MT5_SERVER = os.getenv("MT5_SERVER", "your_server")  # Replace with your server name

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")  # Replace with your bot token
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")  # Replace with your chat ID

# Trading Parameters
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5  # 5-minute timeframe
HIGHER_TIMEFRAME = mt5.TIMEFRAME_M15 # 15-minute timeframe for trend bias

# Timeframe mapping to seconds for sleep calculation
TIMEFRAME_SECONDS = {
    mt5.TIMEFRAME_M1: 60,
    mt5.TIMEFRAME_M5: 300,
    mt5.TIMEFRAME_M15: 900,
    mt5.TIMEFRAME_H1: 3600,
    mt5.TIMEFRAME_H4: 14400,
    mt5.TIMEFRAME_D1: 86400,
}
FAST_MA_PERIOD = 12
SLOW_MA_PERIOD = 50
POSITION_SIZE = 0.01  # Lot size
RISK_REWARD_RATIO = 2.0 # for Take Profit calculation

# --- Function Definitions ---

def initialize_mt5():
    """Initializes and connects to the MetaTrader 5 terminal."""
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False

    # Attempt to login
    authorized = mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
    if not authorized:
        print(f"Failed to connect to account #{MT5_LOGIN}, error code: {mt5.last_error()}")
        mt5.shutdown()
        return False

    print("MetaTrader 5 connection successful.")
    print(f"Connected to account: {mt5.account_info().login}")
    return True

def get_price_data(symbol, timeframe, count):
    """Fetches historical price data from MT5."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            print(f"Failed to get rates for {symbol}, error: {mt5.last_error()}")
            return None

        # Create DataFrame
        df = pd.DataFrame(rates)
        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')

        return df
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return None

def calculate_heikin_ashi(df):
    """Converts regular OHLC data to Heikin Ashi candlesticks."""
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # Calculate initial HA Open
    ha_df.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2

    # Loop to calculate the rest of HA Open values
    for i in range(1, len(ha_df)):
        ha_df.loc[i, 'ha_open'] = (ha_df.loc[i-1, 'ha_open'] + ha_df.loc[i-1, 'ha_close']) / 2

    ha_df['ha_high'] = ha_df[['ha_open', 'ha_close', 'high']].max(axis=1)
    ha_df['ha_low'] = ha_df[['ha_open', 'ha_close', 'low']].min(axis=1)

    # Return a clean dataframe with Heikin Ashi values
    return ha_df[['time', 'ha_open', 'ha_high', 'ha_low', 'ha_close']].rename(
        columns={'ha_open': 'open', 'ha_high': 'high', 'ha_low': 'low', 'ha_close': 'close'}
    )

def calculate_indicators(df):
    """Calculates moving averages on the dataframe."""
    df['fast_ma'] = df['close'].rolling(window=FAST_MA_PERIOD).mean()
    df['slow_ma'] = df['close'].rolling(window=SLOW_MA_PERIOD).mean()
    return df

# A simple state variable to track the last signal to avoid duplicates
last_signal_time = None

def check_trend_bias(symbol, timeframe):
    """Checks the overall trend on a higher timeframe."""
    print("Checking trend bias on", timeframe)
    # Fetch and process data for the higher timeframe
    htf_rates = get_price_data(symbol, timeframe, count=100)
    if htf_rates is None or htf_rates.empty:
        return "NEUTRAL"

    htf_ha = calculate_heikin_ashi(htf_rates)
    htf_ha = calculate_indicators(htf_ha)

    # Check if we have enough data
    if len(htf_ha) < SLOW_MA_PERIOD:
        return "NEUTRAL"

    # Get the last closed candle
    last_candle = htf_ha.iloc[-2]

    if last_candle['fast_ma'] > last_candle['slow_ma']:
        return "BULLISH"
    elif last_candle['fast_ma'] < last_candle['slow_ma']:
        return "BEARISH"
    else:
        return "NEUTRAL"

def check_signals(ha_df, rates_df, trend_bias):
    """Checks for buy or sell signals and calculates SL/TP."""
    global last_signal_time

    print("Checking for signals...")
    if len(ha_df) < SLOW_MA_PERIOD + 2 or len(rates_df) < SLOW_MA_PERIOD + 2:
        print("Not enough data to check for signals.")
        return

    prev_ha_candle = ha_df.iloc[-3]
    last_closed_ha_candle = ha_df.iloc[-2]
    last_closed_raw_candle = rates_df.iloc[-2]

    pip_size = mt5.symbol_info(SYMBOL).point
    sl_buffer = 10 * pip_size

    # --- Buy Signal ---
    is_bullish_cross = prev_ha_candle['fast_ma'] <= prev_ha_candle['slow_ma'] and \
                       last_closed_ha_candle['fast_ma'] > last_closed_ha_candle['slow_ma']

    is_ha_green = last_closed_ha_candle['close'] > last_closed_ha_candle['open']

    if is_bullish_cross and is_ha_green and trend_bias == "BULLISH":
        if last_closed_ha_candle['time'] != last_signal_time:
            signal_price = last_closed_raw_candle['close']

            # SL is the lower of the raw candle's low or the slow MA, plus a buffer
            sl_price = min(last_closed_raw_candle['low'], last_closed_ha_candle['slow_ma']) - sl_buffer

            # TP is based on Risk/Reward ratio
            risk = signal_price - sl_price
            tp_price = signal_price + (risk * RISK_REWARD_RATIO)

            message = (f"📈 BUY SIGNAL\n"
                       f"Symbol: {SYMBOL}\n"
                       f"Entry Price: ~{signal_price:.5f}\n"
                       f"Stop Loss: {sl_price:.5f}\n"
                       f"Take Profit: {tp_price:.5f}\n"
                       f"Time: {last_closed_ha_candle['time']}")
            print(message)
            send_telegram_message(message)
            last_signal_time = last_closed_ha_candle['time']

    # --- Sell Signal ---
    is_bearish_cross = prev_ha_candle['fast_ma'] >= prev_ha_candle['slow_ma'] and \
                       last_closed_ha_candle['fast_ma'] < last_closed_ha_candle['slow_ma']

    is_ha_red = last_closed_ha_candle['close'] < last_closed_ha_candle['open']

    if is_bearish_cross and is_ha_red and trend_bias == "BEARISH":
        if last_closed_ha_candle['time'] != last_signal_time:
            signal_price = last_closed_raw_candle['close']

            # SL is the higher of the raw candle's high or the slow MA, plus a buffer
            sl_price = max(last_closed_raw_candle['high'], last_closed_ha_candle['slow_ma']) + sl_buffer

            # TP is based on Risk/Reward ratio
            risk = sl_price - signal_price
            tp_price = signal_price - (risk * RISK_REWARD_RATIO)

            message = (f"📉 SELL SIGNAL\n"
                       f"Symbol: {SYMBOL}\n"
                       f"Entry Price: ~{signal_price:.5f}\n"
                       f"Stop Loss: {sl_price:.5f}\n"
                       f"Take Profit: {tp_price:.5f}\n"
                       f"Time: {last_closed_ha_candle['time']}")
            print(message)
            send_telegram_message(message)
            last_signal_time = last_closed_ha_candle['time']

def send_telegram_message(message):
    """Sends a message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("Telegram bot token not configured. Skipping message.")
        return
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        print("Telegram message sent successfully.")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def run_strategy():
    """The main function to run the trading strategy."""
    print("Starting trading strategy...")

    if not initialize_mt5():
        return # Stop if connection fails

    # Main loop
    while True:
        try:
            # 1. Get trend from higher timeframe
            trend_bias = check_trend_bias(SYMBOL, HIGHER_TIMEFRAME)

            # 2. Get data for the trading timeframe
            rates_df = get_price_data(SYMBOL, TIMEFRAME, count=100)
            if rates_df is None or rates_df.empty:
                time.sleep(60)
                continue

            # 3. Calculate Heikin Ashi and indicators
            ha_df = calculate_heikin_ashi(rates_df.copy())
            ha_df = calculate_indicators(ha_df)

            # 4. Check for signals
            check_signals(ha_df, rates_df, trend_bias)

            # 5. Wait for the next candle
            timeframe_duration = TIMEFRAME_SECONDS.get(TIMEFRAME)
            if timeframe_duration is None:
                print(f"Unsupported timeframe for sleep calculation: {TIMEFRAME}. Defaulting to 5 minutes.")
                timeframe_duration = 300

            current_time = time.time()
            sleep_time = timeframe_duration - (current_time % timeframe_duration) + 2 # Add 2s buffer
            print(f"Waiting for {int(sleep_time)} seconds until the next candle...")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"An error occurred: {e}")
            send_telegram_message(f"Bot encountered an error: {e}")
            time.sleep(60)

# --- Main Execution ---
if __name__ == "__main__":
    # For security, it's better to use environment variables for credentials
    # You can set them in your OS or use a .env file with a library like python-dotenv
    # For simplicity, we've included placeholders above.
    # Be sure to replace them or set up your environment variables.

    run_strategy()
