import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext
import time
import os
import logging

# --- Configuration ---
# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# MetaTrader 5 Credentials
MT5_LOGIN = int(os.getenv("MT5_LOGIN", 123456))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "your_password")
MT5_SERVER = os.getenv("MT5_SERVER", "your_server")

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")

# Trading Parameters
TIMEFRAME = mt5.TIMEFRAME_M5
HIGHER_TIMEFRAME = mt5.TIMEFRAME_M15
FAST_MA_PERIOD = 12
SLOW_MA_PERIOD = 50
RISK_REWARD_RATIO = 2.0

# A simple in-memory store for user's monitored pairs
# {chat_id: [list_of_symbols]}
monitored_pairs = {}
# Tracks the timestamp of the last signal candle for each symbol to avoid duplicates
# { 'SYMBOL': timestamp }
last_signal_timestamps = {}

# --- Function Definitions ---

def initialize_mt5():
    """Initializes and connects to the MetaTrader 5 terminal."""
    if not mt5.initialize():
        logger.error("initialize() failed, error code = %s", mt5.last_error())
        return False
    try:
        authorized = mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
        if not authorized:
            logger.error("Failed to connect to account #%d, error code: %s", MT5_LOGIN, mt5.last_error())
            mt5.shutdown()
            return False
        logger.info("MetaTrader 5 connection successful to account %d.", mt5.account_info().login)
        return True
    except Exception as e:
        logger.error(f"Exception during MT5 login: {e}")
        return False


def get_price_data(symbol, timeframe, count):
    """Fetches historical price data from MT5."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            logger.warning("Failed to get rates for %s, error: %s", symbol, mt5.last_error())
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logger.error("Error fetching price data for %s: %s", symbol, e)
        return None

def calculate_heikin_ashi(df):
    """Converts regular OHLC data to Heikin Ashi candlesticks."""
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2
    for i in range(1, len(ha_df)):
        ha_df.loc[i, 'ha_open'] = (ha_df.loc[i-1, 'ha_open'] + ha_df.loc[i-1, 'ha_close']) / 2
    ha_df['ha_high'] = ha_df[['ha_open', 'ha_close', 'high']].max(axis=1)
    ha_df['ha_low'] = ha_df[['ha_open', 'ha_close', 'low']].min(axis=1)
    return ha_df[['time', 'ha_open', 'ha_high', 'ha_low', 'ha_close']].rename(
        columns={'ha_open': 'open', 'ha_high': 'high', 'ha_low': 'low', 'ha_close': 'close'}
    )

def calculate_indicators(df):
    """Calculates moving averages on the dataframe."""
    df['fast_ma'] = df['close'].rolling(window=FAST_MA_PERIOD).mean()
    df['slow_ma'] = df['close'].rolling(window=SLOW_MA_PERIOD).mean()
    return df

def check_trend_bias(symbol, timeframe):
    """Checks the overall trend on a higher timeframe."""
    htf_rates = get_price_data(symbol, timeframe, count=100)
    if htf_rates is None or htf_rates.empty:
        return "NEUTRAL"
    htf_ha = calculate_heikin_ashi(htf_rates)
    htf_ha = calculate_indicators(htf_ha)
    if len(htf_ha) < SLOW_MA_PERIOD:
        return "NEUTRAL"
    last_candle = htf_ha.iloc[-2]
    if last_candle['fast_ma'] > last_candle['slow_ma']:
        return "BULLISH"
    elif last_candle['fast_ma'] < last_candle['slow_ma']:
        return "BEARISH"
    else:
        return "NEUTRAL"

def check_and_get_signal(symbol):
    """
    Performs the core signal analysis for a symbol.
    Returns a dictionary with signal details if a new signal is found, otherwise None.
    """
    trend_bias = check_trend_bias(symbol, HIGHER_TIMEFRAME)
    rates_df = get_price_data(symbol, TIMEFRAME, count=100)
    if rates_df is None or rates_df.empty or len(rates_df) < SLOW_MA_PERIOD + 2:
        logger.warning(f"Not enough data to analyze {symbol} for a signal.")
        return None

    ha_df = calculate_heikin_ashi(rates_df.copy())
    ha_df = calculate_indicators(ha_df)

    prev_ha_candle = ha_df.iloc[-3]
    last_closed_ha_candle = ha_df.iloc[-2]
    last_closed_raw_candle = rates_df.iloc[-2]

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.warning(f"Could not get info for {symbol}")
        return None
    pip_size = symbol_info.point
    sl_buffer = 10 * pip_size

    # --- Buy Signal ---
    is_bullish_cross = (prev_ha_candle['fast_ma'] <= prev_ha_candle['slow_ma'] and
                       last_closed_ha_candle['fast_ma'] > last_closed_ha_candle['slow_ma'])
    is_ha_green = last_closed_ha_candle['close'] > last_closed_ha_candle['open']

    if is_bullish_cross and is_ha_green and trend_bias == "BULLISH":
        if last_signal_timestamps.get(symbol) != last_closed_ha_candle['time']:
            last_signal_timestamps[symbol] = last_closed_ha_candle['time']
            signal_price = last_closed_raw_candle['close']
            sl_price = min(last_closed_raw_candle['low'], last_closed_ha_candle['slow_ma']) - sl_buffer
            risk = signal_price - sl_price
            tp_price = signal_price + (risk * RISK_REWARD_RATIO)
            return {'type': 'BUY', 'symbol': symbol, 'price': signal_price, 'sl': sl_price, 'tp': tp_price, 'time': last_closed_ha_candle['time']}

    # --- Sell Signal ---
    is_bearish_cross = (prev_ha_candle['fast_ma'] >= prev_ha_candle['slow_ma'] and
                       last_closed_ha_candle['fast_ma'] < last_closed_ha_candle['slow_ma'])
    is_ha_red = last_closed_ha_candle['close'] < last_closed_ha_candle['open']

    if is_bearish_cross and is_ha_red and trend_bias == "BEARISH":
        if last_signal_timestamps.get(symbol) != last_closed_ha_candle['time']:
            last_signal_timestamps[symbol] = last_closed_ha_candle['time']
            signal_price = last_closed_raw_candle['close']
            sl_price = max(last_closed_raw_candle['high'], last_closed_ha_candle['slow_ma']) + sl_buffer
            risk = sl_price - signal_price
            tp_price = signal_price - (risk * RISK_REWARD_RATIO)
            return {'type': 'SELL', 'symbol': symbol, 'price': signal_price, 'sl': sl_price, 'tp': tp_price, 'time': last_closed_ha_candle['time']}

    return None # No new signal found


def format_signal_message(signal_dict: dict) -> str:
    """Formats a signal dictionary into a user-friendly string."""
    icon = "📈" if signal_dict['type'] == 'BUY' else "📉"
    return (f"{icon} {signal_dict['type']} SIGNAL\n"
            f"Symbol: {signal_dict['symbol']}\n"
            f"Entry Price: ~{signal_dict['price']:.5f}\n"
            f"Stop Loss: {signal_dict['sl']:.5f}\n"
            f"Take Profit: {signal_dict['tp']:.5f}\n"
            f"Time: {signal_dict['time']}")


def get_signal_for_symbol(symbol: str) -> str:
    """
    Wrapper function for the /signal command.
    Analyzes a single symbol and returns a signal message or a 'no signal' message.
    """
    # Ensure MT5 is connected before proceeding
    if not mt5.terminal_state().connected:
        if not initialize_mt5():
            return "Error: Could not connect to the trading server. Please try again later."

    signal = check_and_get_signal(symbol)
    if signal:
        return format_signal_message(signal)
    else:
        # Check for data availability to give a more informative message
        rates_df = get_price_data(symbol, TIMEFRAME, count=2)
        if rates_df is None or rates_df.empty:
            return f"Could not retrieve data for {symbol}. It might be an invalid symbol."
        return f"No clear signal for {symbol} at the moment."

# --- Telegram Command Handlers ---

def start(update: Update, context: CallbackContext) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    welcome_message = (
        f"👋 Hello {user.first_name}!\n\n"
        "I am your Forex Signal Bot. Here's how you can use me:\n\n"
        "🔹 /signal - Get an immediate signal for a specific currency pair.\n"
        "🔹 /monitor - Choose pairs to monitor continuously for signals.\n"
        "🔹 /unmonitor - Stop monitoring specific pairs.\n"
        "🔹 /monitoring - See the list of pairs you are currently monitoring.\n"
        "🔹 /help - Show this help message again.\n\n"
        "Let's get started!"
    )
    update.message.reply_text(welcome_message)

def help_command(update: Update, context: CallbackContext) -> None:
    """Sends help message."""
    start(update, context)


def signal_command(update: Update, context: CallbackContext) -> None:
    """Displays an inline keyboard for the user to select a symbol for a one-time signal."""
    keyboard = [
        [
            InlineKeyboardButton("EURUSD", callback_data='signal_EURUSD'),
            InlineKeyboardButton("GBPUSD", callback_data='signal_GBPUSD'),
        ],
        [
            InlineKeyboardButton("USDJPY", callback_data='signal_USDJPY'),
            InlineKeyboardButton("AUDUSD", callback_data='signal_AUDUSD'),
        ],
        [
            InlineKeyboardButton("USDCAD", callback_data='signal_USDCAD'),
            InlineKeyboardButton("XAUUSD", callback_data='signal_XAUUSD'),  # Gold
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Please choose a symbol to get a signal for:', reply_markup=reply_markup)


def monitor_command(update: Update, context: CallbackContext) -> None:
    """Displays an inline keyboard for the user to select symbols to monitor."""
    keyboard = [
        [
            InlineKeyboardButton("EURUSD", callback_data='monitor_EURUSD'),
            InlineKeyboardButton("GBPUSD", callback_data='monitor_GBPUSD'),
        ],
        [
            InlineKeyboardButton("USDJPY", callback_data='monitor_USDJPY'),
            InlineKeyboardButton("AUDUSD", callback_data='monitor_AUDUSD'),
        ],
        [
            InlineKeyboardButton("USDCAD", callback_data='monitor_USDCAD'),
            InlineKeyboardButton("XAUUSD", callback_data='monitor_XAUUSD'),
        ],
        [InlineKeyboardButton("Done", callback_data='monitor_done')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Choose symbols to add to your monitoring list:', reply_markup=reply_markup)


def unmonitor_command(update: Update, context: CallbackContext) -> None:
    """Displays an inline keyboard of currently monitored symbols for the user to remove."""
    chat_id = update.message.chat_id
    user_monitored_pairs = monitored_pairs.get(chat_id, [])

    if not user_monitored_pairs:
        update.message.reply_text("You are not monitoring any pairs yet. Use /monitor to add some.")
        return

    keyboard = [[InlineKeyboardButton(symbol, callback_data=f'unmonitor_{symbol}')] for symbol in user_monitored_pairs]
    keyboard.append([InlineKeyboardButton("Done", callback_data='unmonitor_done')])

    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Choose symbols to remove from your monitoring list:', reply_markup=reply_markup)


def monitoring_command(update: Update, context: CallbackContext) -> None:
    """Lists the symbols the user is currently monitoring."""
    chat_id = update.message.chat_id
    user_monitored_pairs = monitored_pairs.get(chat_id, [])

    if not user_monitored_pairs:
        message = "You are not monitoring any pairs yet. Use /monitor to add some."
    else:
        message = "You are currently monitoring the following pairs:\n" + "\n".join(f"• {s}" for s in user_monitored_pairs)

    update.message.reply_text(message)


def button_callback(update: Update, context: CallbackContext) -> None:
    """Parses the CallbackQuery and runs the appropriate action."""
    query = update.callback_query
    query.answer()  # Acknowledge the button press

    action, *data = query.data.split('_', 1)
    symbol = data[0] if data else None
    chat_id = query.message.chat_id

    if action == 'signal':
        query.edit_message_text(text=f"⏳ Analyzing {symbol}, please wait...")
        signal_message = get_signal_for_symbol(symbol)
        query.edit_message_text(text=signal_message)

    elif action == 'monitor':
        if symbol == 'done':
            query.edit_message_text(text="Your monitoring list has been updated.")
            return

        # Initialize list if not present
        if chat_id not in monitored_pairs:
            monitored_pairs[chat_id] = []

        # Add symbol if not already there
        if symbol not in monitored_pairs[chat_id]:
            monitored_pairs[chat_id].append(symbol)
            query.answer(text=f"✅ Added {symbol} to your monitoring list.")
        else:
            query.answer(text=f"ℹ️ You are already monitoring {symbol}.")

    elif action == 'unmonitor':
        if symbol == 'done':
            query.edit_message_text(text="Your monitoring list has been updated.")
            return

        if chat_id in monitored_pairs and symbol in monitored_pairs[chat_id]:
            monitored_pairs[chat_id].remove(symbol)
            query.answer(text=f"❌ Removed {symbol} from your monitoring list.")
            # Refresh the keyboard
            user_monitored_pairs = monitored_pairs.get(chat_id, [])
            if not user_monitored_pairs:
                query.edit_message_text("You are no longer monitoring any pairs.")
            else:
                keyboard = [[InlineKeyboardButton(s, callback_data=f'unmonitor_{s}')] for s in user_monitored_pairs]
                keyboard.append([InlineKeyboardButton("Done", callback_data='unmonitor_done')])
                reply_markup = InlineKeyboardMarkup(keyboard)
                query.edit_message_text('Choose symbols to remove:', reply_markup=reply_markup)
        else:
            query.answer(text=f"ℹ️ You are not monitoring {symbol}.")

    else:
        query.edit_message_text(text=f"Unknown action: {action}")


def check_all_monitored_pairs_job(context: CallbackContext) -> None:
    """
    This job iterates through all monitored pairs and sends a signal if one is found.
    """
    if not mt5.terminal_state().connected:
        logger.warning("Job running, but MT5 is not connected. Attempting to reconnect.")
        if not initialize_mt5():
            logger.error("Job failed to reconnect to MT5. Skipping this run.")
            return

    logger.info("Running scheduled job: Checking all monitored pairs...")

    # Create a copy of the items to avoid issues if the dict is modified during iteration
    for chat_id, symbols in list(monitored_pairs.items()):
        for symbol in symbols:
            try:
                signal = check_and_get_signal(symbol)
                if signal:
                    message = format_signal_message(signal)
                    # Send the message to the user
                    context.bot.send_message(chat_id=chat_id, text=message)
                    logger.info(f"Sent signal for {symbol} to chat_id {chat_id}")
            except Exception as e:
                logger.error(f"Error checking signal for {symbol} for chat_id {chat_id}: {e}")


# --- Main Bot Logic ---

def main() -> None:
    """Start the telegram bot."""
    # Initialize MT5
    if not initialize_mt5():
        logger.error("Failed to initialize MetaTrader 5. The bot will not be able to fetch data.")
        # Decide if you want the bot to run without MT5 connection.
        # For this example, we'll let it run to allow user interaction,
        # but signal-related commands will fail.

    # Create the Updater and pass it your bot's token.
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("Telegram Bot Token is not configured. Please set the TELEGRAM_BOT_TOKEN environment variable.")
        return

    updater = Updater(TELEGRAM_BOT_TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("signal", signal_command))
    dispatcher.add_handler(CommandHandler("monitor", monitor_command))
    dispatcher.add_handler(CommandHandler("unmonitor", unmonitor_command))
    dispatcher.add_handler(CommandHandler("monitoring", monitoring_command))
    # This handler will process all button clicks from inline keyboards
    dispatcher.add_handler(CallbackQueryHandler(button_callback))

    # Schedule the monitoring job
    job_queue = updater.job_queue
    # Run the job every 5 minutes (300 seconds)
    job_queue.run_repeating(check_all_monitored_pairs_job, interval=300, first=10)
    logger.info("Scheduled monitoring job to run every 5 minutes.")

    # Start the Bot
    updater.start_polling()
    logger.info("Bot has started successfully.")

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

    # Shutdown MT5 connection when the bot is stopped
    mt5.shutdown()
    logger.info("MetaTrader 5 connection shut down.")

if __name__ == '__main__':
    main()
