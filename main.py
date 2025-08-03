import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackQueryHandler,
    CallbackContext,
    ConversationHandler,
    MessageHandler,
    Filters,
)
import time
import os
import logging

# States for conversation
(
    CHOOSING_SIGNAL, TYPING_CUSTOM_SIGNAL,
    ASKING_LOT_SIZE,
) = range(3)


# --- Configuration ---
# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# MetaTrader 5 Credentials
MT5_LOGIN = 8930488
MT5_PASSWORD = "LiusPro123_"
MT5_SERVER = "HFMarketsGlobal-Demo"

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "8040654533:AAEMZmRFP75DbfhclicAI8HHtgsGqgJVIZ8"

# Trading Parameters
TIMEFRAME = mt5.TIMEFRAME_M5
FAST_MA_PERIOD = 12
SLOW_MA_PERIOD = 50
RISK_REWARD_RATIO = 2.0

# A simple in-memory store for user's monitored pairs
# {chat_id: [list_of_symbols]}
monitored_pairs = {}
# Tracks the timestamp of the last signal candle for each symbol to avoid duplicates
# { 'SYMBOL': timestamp }
last_signal_timestamps = {}
# {chat_id: {'auto_trade': False}}
user_settings = {}

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

def is_resistance_broken(rates_df: pd.DataFrame, lookback: int = 50) -> bool:
    """
    Checks if the last closed candle has broken a recent resistance level.
    Resistance is defined as the highest high in the lookback period before the breakout candle.
    """
    # We look for resistance in the candles *before* the potential breakout.
    # Exclude the last 3 candles to give room for the setup to form.
    search_df = rates_df.iloc[-(lookback + 3):-3]
    if search_df.empty:
        return False

    resistance_level = search_df['high'].max()
    last_closed_price = rates_df.iloc[-2]['close']

    if last_closed_price > resistance_level:
        logger.info(f"Resistance broken: Last close {last_closed_price:.5f} > Resistance {resistance_level:.5f}")
        return True
    return False

def is_support_broken(rates_df: pd.DataFrame, lookback: int = 50) -> bool:
    """
    Checks if the last closed candle has broken a recent support level.
    Support is defined as the lowest low in the lookback period before the breakdown candle.
    """
    # We look for support in the candles *before* the potential breakdown.
    # Exclude the last 3 candles to give room for the setup to form.
    search_df = rates_df.iloc[-(lookback + 3):-3]
    if search_df.empty:
        return False

    support_level = search_df['low'].min()
    last_closed_price = rates_df.iloc[-2]['close']

    if last_closed_price < support_level:
        logger.info(f"Support broken: Last close {last_closed_price:.5f} < Support {support_level:.5f}")
        return True
    return False

def get_exit_strategies(ha_df: pd.DataFrame, signal_type: str) -> dict:
    """
    Calculates suggested exit levels based on dynamic strategies.
    """
    last_closed_ha_candle = ha_df.iloc[-2]
    exits = {}

    # Conservative exit is a condition, not a fixed price.
    exits['conservative_exit'] = "Close if Heikin Ashi candle color flips."

    # Aggressive exit is based on the Fast MA value at the time of the signal.
    aggressive_exit_price = last_closed_ha_candle['fast_ma']
    exits['aggressive_exit'] = f"~{aggressive_exit_price:.5f} (if price crosses 12-MA)"

    return exits

def check_buy_signal(symbol, ha_df, rates_df, pip_size):
    """Checks for a buy signal based on MA cross and HA."""
    prev_ha_candle = ha_df.iloc[-3]
    last_closed_ha_candle = ha_df.iloc[-2]
    last_closed_raw_candle = rates_df.iloc[-2]
    sl_buffer = 10 * pip_size

    is_bullish_cross = (prev_ha_candle['fast_ma'] <= prev_ha_candle['slow_ma'] and
                       last_closed_ha_candle['fast_ma'] > last_closed_ha_candle['slow_ma'])
    is_ha_green = last_closed_ha_candle['close'] > last_closed_ha_candle['open']
    # A strong buy signal candle has no lower wick.
    is_strong_buy_candle = last_closed_ha_candle['open'] == last_closed_ha_candle['low']

    # Breakout is now a point of interest, not a strict condition.
    breakout_poi = is_resistance_broken(rates_df)
    reason = "MA Crossover + Heikin Ashi"
    if breakout_poi:
        reason += " (Resistance Break POI)"


    if is_bullish_cross and is_ha_green and is_strong_buy_candle:
        if last_signal_timestamps.get(symbol) != last_closed_ha_candle['time']:
            last_signal_timestamps[symbol] = last_closed_ha_candle['time']
            signal_price = last_closed_raw_candle['close']
            sl_price = min(last_closed_raw_candle['low'], last_closed_ha_candle['fast_ma'], last_closed_ha_candle['slow_ma']) - sl_buffer
            risk = signal_price - sl_price
            tp_price = signal_price + (risk * RISK_REWARD_RATIO)

            signal_info = {
                'type': 'BUY', 'symbol': symbol, 'price': signal_price,
                'sl': sl_price, 'tp': tp_price, 'time': last_closed_ha_candle['time'],
                'reason': reason
            }
            signal_info['exits'] = get_exit_strategies(ha_df, 'BUY')
            return signal_info
    return None

def check_sell_signal(symbol, ha_df, rates_df, pip_size):
    """Checks for a sell signal based on MA cross and HA."""
    prev_ha_candle = ha_df.iloc[-3]
    last_closed_ha_candle = ha_df.iloc[-2]
    last_closed_raw_candle = rates_df.iloc[-2]
    sl_buffer = 10 * pip_size

    is_bearish_cross = (prev_ha_candle['fast_ma'] >= prev_ha_candle['slow_ma'] and
                       last_closed_ha_candle['fast_ma'] < last_closed_ha_candle['slow_ma'])
    is_ha_red = last_closed_ha_candle['close'] < last_closed_ha_candle['open']
    # A strong sell signal candle has no upper wick.
    is_strong_sell_candle = last_closed_ha_candle['open'] == last_closed_ha_candle['high']

    # Support breakout is a point of interest, not a strict condition
    breakout_poi = is_support_broken(rates_df)
    reason = "MA Crossover + Heikin Ashi"
    if breakout_poi:
        reason += " (Support Break POI)"

    if is_bearish_cross and is_ha_red and is_strong_sell_candle:
        if last_signal_timestamps.get(symbol) != last_closed_ha_candle['time']:
            last_signal_timestamps[symbol] = last_closed_ha_candle['time']
            signal_price = last_closed_raw_candle['close']
            # For a sell, SL is above the high, or MAs
            sl_price = max(last_closed_raw_candle['high'], last_closed_ha_candle['fast_ma'], last_closed_ha_candle['slow_ma']) + sl_buffer
            risk = sl_price - signal_price
            tp_price = signal_price - (risk * RISK_REWARD_RATIO)

            signal_info = {
                'type': 'SELL', 'symbol': symbol, 'price': signal_price,
                'sl': sl_price, 'tp': tp_price, 'time': last_closed_ha_candle['time'],
                'reason': reason
            }
            signal_info['exits'] = get_exit_strategies(ha_df, 'SELL')
            return signal_info
    return None

def check_and_get_signal(symbol):
    """
    Performs the core signal analysis for a symbol by checking for buy and sell signals.
    Returns a dictionary with signal details if a new signal is found, otherwise None.
    """
    # Increased count to have enough data for pattern analysis
    rates_df = get_price_data(symbol, TIMEFRAME, count=200)
    if rates_df is None or rates_df.empty or len(rates_df) < SLOW_MA_PERIOD + 50: # Increased buffer
        logger.warning(f"Not enough data to analyze {symbol} for a signal.")
        return None

    # Calculate indicators on the raw price data first
    rates_df_with_indicators = calculate_indicators(rates_df.copy())

    # Now, calculate Heikin Ashi candles from the original rates_df
    ha_df = calculate_heikin_ashi(rates_df.copy())

    # Add the indicators from the raw data to the Heikin Ashi dataframe
    # This keeps the HA candle structure but uses the correct MA values for checks
    ha_df['fast_ma'] = rates_df_with_indicators['fast_ma']
    ha_df['slow_ma'] = rates_df_with_indicators['slow_ma']


    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.warning(f"Could not get info for {symbol}")
        return None
    pip_size = symbol_info.point

    # Check for buy signal
    buy_signal = check_buy_signal(symbol, ha_df, rates_df, pip_size)
    if buy_signal:
        return buy_signal

    # Check for sell signal
    sell_signal = check_sell_signal(symbol, ha_df, rates_df, pip_size)
    if sell_signal:
        return sell_signal

    return None # No new signal found


def format_signal_message(signal_dict: dict) -> str:
    """Formats a signal dictionary into a user-friendly string."""
    icon = "📈" if signal_dict['type'] == 'BUY' else "📉"

    message = (
        f"{icon} {signal_dict['type']} SIGNAL for {signal_dict['symbol']}\n\n"
        f"Reason: {signal_dict.get('reason', 'N/A')}\n"
        f"------------------------------------\n"
        f"Entry Price: ~{signal_dict['price']:.5f}\n"
        f"Stop Loss: {signal_dict['sl']:.5f}\n"
        f"Take Profit (R:R 2.0): {signal_dict['tp']:.5f}\n"
    )

    if 'exits' in signal_dict:
        message += "\n--- Dynamic Exit Suggestions ---\n"
        if 'conservative_exit' in signal_dict['exits']:
            message += f"🔹 Conservative: {signal_dict['exits']['conservative_exit']}\n"
        if 'aggressive_exit' in signal_dict['exits']:
            message += f"🔸 Aggressive: {signal_dict['exits']['aggressive_exit']}\n"

    message += f"\nSignal Time: {signal_dict['time']}"
    return message


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
        return format_signal_message(signal), signal # Return both message and raw signal
    else:
        # Check for data availability to give a more informative message
        rates_df = get_price_data(symbol, TIMEFRAME, count=2)
        if rates_df is None or rates_df.empty:
            return f"Could not retrieve data for {symbol}. It might be an invalid symbol.", None
        return f"No clear signal for {symbol} at the moment.", None


def place_market_order(signal_type: str, symbol: str, lot_size: float, sl_price: float, tp_price: float) -> str:
    """
    Places a market order on MetaTrader 5.
    Returns a string with the result of the operation.
    """
    logger.info(f"Attempting to place {signal_type} order for {symbol} with lot size {lot_size}.")

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.error(f"Could not get info for {symbol}, cannot place order.")
        return f"❌ **Order Failed**: Could not retrieve symbol information for {symbol}."

    if signal_type == 'BUY':
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif signal_type == 'SELL':
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        logger.warning(f"Invalid order type specified: {signal_type}")
        return f"❌ **Order Failed**: Invalid order type '{signal_type}'."

    # Verify that the prices are properly rounded to the symbol's digits
    sl_price = round(sl_price, symbol_info.digits)
    tp_price = round(tp_price, symbol_info.digits)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20, # Allowed slippage in points
        "magic": 234000, # A magic number to identify orders from this bot
        "comment": "Telegram Signal Bot",
        "type_time": mt5.ORDER_TIME_GTC, # Good till cancelled
        "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel
    }

    try:
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed for {symbol}. Code: {result.retcode}, Comment: {result.comment}")
            return f"❌ **Order Failed** for {symbol}.\nReason: {result.comment} (Code: {result.retcode})"
        else:
            logger.info(f"Order successful for {symbol}. Ticket: {result.order}, Price: {result.price}, Volume: {result.volume}")
            return (f"✅ **Order Successful!**\n\n"
                    f"🔹 **Symbol:** {symbol}\n"
                    f"🔹 **Type:** {signal_type}\n"
                    f"🔹 **Lot Size:** {result.volume}\n"
                    f"🔹 **Ticket:** {result.order}")
    except Exception as e:
        logger.error(f"An exception occurred while placing order for {symbol}: {e}")
        return f"❌ **Order Failed**: An unexpected error occurred: {e}"


# --- Telegram Command Handlers ---

def start(update: Update, context: CallbackContext) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    welcome_message = (
        f"👋 Hello {user.first_name}!\n\n"
        "I am your Forex Signal & Trading Bot. Here's a list of commands:\n\n"
        "**Signal Commands**\n"
        "🔹 /signal - Get an immediate signal for a currency pair.\n"
        "🔹 /monitor - Select pairs to monitor for signals.\n"
        "🔹 /unmonitor - Stop monitoring pairs.\n"
        "🔹 /monitoring - List your monitored pairs.\n\n"
        "**Trading & Account Commands**\n"
        "📈 /status - Check your account balance, equity, and PnL.\n"
        "📉 /pnl - Check the PnL for a specific open order.\n"
        "🤖 /autotrade_on - Enable automatic trading (lot size 0.01).\n"
        "🤖 /autotrade_off - Disable automatic trading.\n"
        "🤖 /autotrade_status - Check if auto-trading is on or off.\n\n"
        "🔹 /help - Show this help message again.\n\n"
        "When a signal is received and auto-trade is off, you will see buttons to manually execute the trade."
    )
    update.message.reply_text(welcome_message)

def help_command(update: Update, context: CallbackContext) -> None:
    """Sends help message."""
    start(update, context)

def autotrade_on(update: Update, context: CallbackContext) -> None:
    """Enables auto-trading for the user."""
    chat_id = update.message.chat_id
    if chat_id not in user_settings:
        user_settings[chat_id] = {}
    user_settings[chat_id]['auto_trade'] = True
    logger.info(f"Auto-trading enabled for chat_id: {chat_id}")
    update.message.reply_text("🤖 Auto-trading has been **ENABLED**. I will now execute trades automatically with a lot size of 0.01.")

def autotrade_off(update: Update, context: CallbackContext) -> None:
    """Disables auto-trading for the user."""
    chat_id = update.message.chat_id
    if chat_id in user_settings:
        user_settings[chat_id]['auto_trade'] = False
    # Also handles cases where the user was never in the settings dict
    else:
        user_settings[chat_id] = {'auto_trade': False}
    logger.info(f"Auto-trading disabled for chat_id: {chat_id}")
    update.message.reply_text("🤖 Auto-trading has been **DISABLED**. I will ask for confirmation before placing trades.")

def autotrade_status(update: Update, context: CallbackContext) -> None:
    """Checks the current auto-trading status for the user."""
    chat_id = update.message.chat_id
    status = user_settings.get(chat_id, {}).get('auto_trade', False)
    if status:
        update.message.reply_text("🤖 Auto-trading is currently **ENABLED**.")
    else:
        update.message.reply_text("🤖 Auto-trading is currently **DISABLED**.")


def account_status(update: Update, context: CallbackContext) -> None:
    """Displays the current MetaTrader account status."""
    if not mt5.terminal_state().connected:
        if not initialize_mt5():
            update.message.reply_text("❌ Could not connect to the trading server. Please try again later.")
            return

    account_info = mt5.account_info()
    if not account_info:
        update.message.reply_text("❌ Failed to retrieve account information.")
        return

    positions = mt5.positions_get()
    if positions is None:
        update.message.reply_text("❌ Failed to retrieve open positions. The connection might have been lost.")
        return

    num_positions = len(positions)
    total_pnl = sum(pos.profit for pos in positions)

    pnl_icon = "🟢" if total_pnl >= 0 else "🔴"
    message = (
        f"📊 **Account Status**\n\n"
        f"🔹 **Balance:** {account_info.balance:.2f} {account_info.currency}\n"
        f"🔹 **Equity:** {account_info.equity:.2f} {account_info.currency}\n"
        f"🔹 **Open Positions:** {num_positions}\n"
        f"{pnl_icon} **Total PnL:** {total_pnl:.2f} {account_info.currency}"
    )

    update.message.reply_text(message, parse_mode='Markdown')


def pnl_command(update: Update, context: CallbackContext) -> None:
    """Starts the conversation to check PnL for a specific order."""
    if not mt5.terminal_state().connected:
        if not initialize_mt5():
            update.message.reply_text("❌ Could not connect to the trading server. Please try again later.")
            return

    positions = mt5.positions_get()
    if positions is None:
        update.message.reply_text("❌ Failed to retrieve open positions. The connection might have been lost.")
        return

    if not positions:
        update.message.reply_text("ℹ️ You have no open orders to check.")
        return

    keyboard = []
    for pos in positions:
        trade_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        button_text = f"{trade_type} {pos.symbol} {pos.volume} lot"
        callback_data = f"pnl_{pos.ticket}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text("Please select an order to check its PnL:", reply_markup=reply_markup)


def pnl_callback(update: Update, context: CallbackContext) -> None:
    """Handles the PnL button press and shows details for a specific order."""
    query = update.callback_query
    query.answer()

    try:
        _, ticket_str = query.data.split('_')
        ticket = int(ticket_str)
    except (ValueError, IndexError):
        query.edit_message_text("❌ Invalid callback data.")
        return

    if not mt5.terminal_state().connected:
        if not initialize_mt5():
            query.edit_message_text("❌ Could not connect to the trading server.")
            return

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        query.edit_message_text(f"❌ Could not find order with ticket {ticket}. It might have been closed.")
        return

    pos = positions[0]
    account_info = mt5.account_info()
    currency = account_info.currency if account_info else ""

    trade_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
    pnl_icon = "🟢" if pos.profit >= 0 else "🔴"

    message = (
        f"🔍 **PnL for Ticket {pos.ticket}**\n\n"
        f"🔹 **Symbol:** {pos.symbol}\n"
        f"🔹 **Type:** {trade_type}\n"
        f"🔹 **Volume:** {pos.volume} lot\n"
        f"------------------------------------\n"
        f"🔹 **Open Price:** {pos.price_open:.5f}\n"
        f"🔹 **Current Price:** {pos.price_current:.5f}\n"
        f"🔹 **Stop Loss:** {pos.sl:.5f}\n"
        f"🔹 **Take Profit:** {pos.tp:.5f}\n"
        f"------------------------------------\n"
        f"{pnl_icon} **Profit/Loss:** {pos.profit:.2f} {currency}"
    )

    query.edit_message_text(text=message, parse_mode='Markdown')


def trade_callback(update: Update, context: CallbackContext) -> int:
    """Handles the 'BUY' or 'SELL' button press to start the trade conversation."""
    query = update.callback_query
    query.answer()

    # e.g., 'trade_buy_EURUSD'
    _, trade_type, symbol = query.data.split('_')

    # It's crucial to fetch the latest signal data right before the trade
    # to ensure prices (SL/TP) are as relevant as possible.
    signal = check_and_get_signal(symbol)

    if not signal or signal['type'].lower() != trade_type:
        query.edit_message_text(text="⚠️ The signal has expired or conditions have changed. Please request a new signal analysis.")
        return ConversationHandler.END

    # Store the validated signal data in user_data for the next step
    context.user_data['trade_signal'] = signal
    logger.info(f"User {query.from_user.id} initiated {trade_type} for {symbol}. Asking for lot size.")

    # Edit the original message to ask for lot size
    query.edit_message_text(
        text=f"You've initiated a **{trade_type.upper()}** for **{symbol}**.\n\n"
             f"Please enter the lot size you wish to use (e.g., `0.01`, `0.1`).\n\n"
             f"Type /cancel to abort."
    )

    return ASKING_LOT_SIZE


def lot_size_input(update: Update, context: CallbackContext) -> int:
    """Receives lot size, places the order, and ends the conversation."""
    lot_size_str = update.message.text
    signal = context.user_data.get('trade_signal')

    if not signal:
        update.message.reply_text("❌ An error occurred: I've lost the context of the trade. Please start over by requesting a new signal.")
        return ConversationHandler.END

    try:
        # Validate the lot size
        lot_size = float(lot_size_str)
        if lot_size <= 0:
            raise ValueError("Lot size must be a positive number.")
    except ValueError:
        update.message.reply_text(
            "⚠️ Invalid lot size. Please enter a positive number (e.g., `0.01`).\n\n"
            "Or type /cancel to abort."
        )
        return ASKING_LOT_SIZE # Ask again without ending the conversation

    logger.info(f"User {update.effective_user.id} entered lot size {lot_size}. Placing order.")
    update.message.reply_text(f"⏳ Understood. Placing a **{signal['type']}** order for **{signal['symbol']}** with lot size **{lot_size}**...")

    # Execute the trade
    trade_result = place_market_order(
        signal_type=signal['type'],
        symbol=signal['symbol'],
        lot_size=lot_size,
        sl_price=signal['sl'],
        tp_price=signal['tp']
    )

    update.message.reply_text(trade_result, parse_mode='Markdown')

    # Clean up the user_data to free memory
    if 'trade_signal' in context.user_data:
        del context.user_data['trade_signal']

    return ConversationHandler.END


def signal_command(update: Update, context: CallbackContext) -> int:
    """Displays an inline keyboard for the user to select a symbol for a one-time signal. Starts the conversation."""
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
        [InlineKeyboardButton("Custom Pair", callback_data='signal_custom')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Please choose a symbol to get a signal for, or choose "Custom":', reply_markup=reply_markup)
    return CHOOSING_SIGNAL


def signal_button_callback(update: Update, context: CallbackContext) -> int:
    """Handles button presses for the signal command, gets signal, and shows trade buttons."""
    query = update.callback_query
    query.answer()

    action, *data = query.data.split('_', 1)
    symbol = data[0] if data else None

    if symbol == 'custom':
        query.edit_message_text(text="Please send me the symbol you want to analyze (e.g., EURUSD, BTCUSD).")
        return TYPING_CUSTOM_SIGNAL

    if symbol:
        query.edit_message_text(text=f"⏳ Analyzing {symbol}, please wait...")
        # get_signal_for_symbol now returns (message, signal_object)
        signal_message, signal_data = get_signal_for_symbol(symbol)

        if signal_data:
            # If there's a signal, show the trade buttons
            keyboard = [
                [
                    InlineKeyboardButton(f"📈 BUY {symbol}", callback_data=f"trade_buy_{symbol}"),
                    InlineKeyboardButton(f"📉 SELL {symbol}", callback_data=f"trade_sell_{symbol}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            query.edit_message_text(text=signal_message, reply_markup=reply_markup)
        else:
            # If no signal, just show the message
            query.edit_message_text(text=signal_message)

    return ConversationHandler.END


def custom_signal_input(update: Update, context: CallbackContext) -> int:
    """Handles custom symbol input, gets signal, and shows trade buttons."""
    symbol = update.message.text.upper()

    update.message.reply_text(f"⏳ Analyzing {symbol}, please wait...")
    signal_message, signal_data = get_signal_for_symbol(symbol)

    if signal_data:
        # If there's a signal, show the trade buttons
        keyboard = [
            [
                InlineKeyboardButton(f"📈 BUY {symbol}", callback_data=f"trade_buy_{symbol}"),
                InlineKeyboardButton(f"📉 SELL {symbol}", callback_data=f"trade_sell_{symbol}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text(text=signal_message, reply_markup=reply_markup)
    else:
        # If no signal, just show the message
        update.message.reply_text(text=signal_message)

    return ConversationHandler.END


def cancel_conversation(update: Update, context: CallbackContext) -> int:
    """Cancels and ends the conversation."""
    update.message.reply_text('Operation cancelled.')
    return ConversationHandler.END


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

    if action == 'monitor':
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
    This job iterates through all monitored pairs, checks for signals, and then
    either executes a trade automatically or sends a notification with options.
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
                    # Check if the user has auto-trading enabled
                    is_auto_trade = user_settings.get(chat_id, {}).get('auto_trade', False)

                    if is_auto_trade:
                        logger.info(f"Auto-trading is ON for chat_id {chat_id}. Placing trade for {symbol}.")
                        trade_result = place_market_order(
                            signal_type=signal['type'],
                            symbol=signal['symbol'],
                            lot_size=0.01,
                            sl_price=signal['sl'],
                            tp_price=signal['tp']
                        )
                        # Notify the user about the auto-trade
                        auto_trade_message = f"🤖 **Auto-Trade Executed** for {symbol}.\n\n{trade_result}"
                        context.bot.send_message(chat_id=chat_id, text=auto_trade_message)
                    else:
                        # Manual trade: send signal with action buttons
                        message = format_signal_message(signal)
                        keyboard = [
                            [
                                InlineKeyboardButton(f"📈 BUY {symbol}", callback_data=f"trade_buy_{symbol}"),
                                InlineKeyboardButton(f"📉 SELL {symbol}", callback_data=f"trade_sell_{symbol}")
                            ]
                        ]
                        reply_markup = InlineKeyboardMarkup(keyboard)
                        context.bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)

                    logger.info(f"Sent signal for {symbol} to chat_id {chat_id} (Auto-Trade: {is_auto_trade})")

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

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add command handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("monitor", monitor_command))
    dispatcher.add_handler(CommandHandler("unmonitor", unmonitor_command))
    dispatcher.add_handler(CommandHandler("monitoring", monitoring_command))
    dispatcher.add_handler(CommandHandler("autotrade_on", autotrade_on))
    dispatcher.add_handler(CommandHandler("autotrade_off", autotrade_off))
    dispatcher.add_handler(CommandHandler("autotrade_status", autotrade_status))
    dispatcher.add_handler(CommandHandler("status", account_status))
    dispatcher.add_handler(CommandHandler("pnl", pnl_command))

    # Set up the conversation handler for the /signal command
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('signal', signal_command)],
        states={
            CHOOSING_SIGNAL: [CallbackQueryHandler(signal_button_callback, pattern='^signal_')],
            TYPING_CUSTOM_SIGNAL: [MessageHandler(Filters.text & ~Filters.command, custom_signal_input)],
        },
        fallbacks=[CommandHandler('cancel', cancel_conversation)],
        per_message=False
    )
    dispatcher.add_handler(conv_handler)

    # This handler processes button clicks for monitoring
    dispatcher.add_handler(CallbackQueryHandler(button_callback, pattern='^(monitor|unmonitor)_'))

    # This handler processes PnL button clicks
    dispatcher.add_handler(CallbackQueryHandler(pnl_callback, pattern='^pnl_'))

    # Set up the conversation handler for placing trades
    trade_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(trade_callback, pattern='^trade_')],
        states={
            ASKING_LOT_SIZE: [MessageHandler(Filters.text & ~Filters.command, lot_size_input)],
        },
        fallbacks=[CommandHandler('cancel', cancel_conversation)],
        per_message=False # Make sure conversation is on a per-user basis
    )
    dispatcher.add_handler(trade_conv_handler)

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
