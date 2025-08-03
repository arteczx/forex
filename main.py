import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)
import time
import os
import logging
import asyncio

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
MT5_LOGIN = 48930488
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
# {chat_id: {'auto_trade': False, 'mode': 'ha_ma'}}
user_settings = {}
# {ticket_id: chat_id} to notify users of auto-closures
trade_to_chat_id = {}


# --- Function Definitions ---

def get_user_settings(chat_id: int) -> dict:
    """Gets or creates the settings for a user, with defaults."""
    if chat_id not in user_settings:
        user_settings[chat_id] = {
            'auto_trade': False,
            'mode': 'ha_ma'  # Default mode
        }
    # Ensure 'mode' exists for older users who might be in the dictionary already
    elif 'mode' not in user_settings[chat_id]:
        user_settings[chat_id]['mode'] = 'ha_ma'
    return user_settings[chat_id]


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
                'reason': reason,
                'mode': 'ha_ma'  # Identify signal mode
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
                'reason': reason,
                'mode': 'ha_ma'  # Identify signal mode
            }
            signal_info['exits'] = get_exit_strategies(ha_df, 'SELL')
            return signal_info
    return None


def check_buy_signal_pure_ha(symbol, ha_df, rates_df, pip_size):
    """Checks for a buy signal based on pure Heikin Ashi reversal."""
    if len(ha_df) < 3:  # Need at least previous and last closed candle
        return None

    prev_ha_candle = ha_df.iloc[-3]
    last_closed_ha_candle = ha_df.iloc[-2]
    last_closed_raw_candle = rates_df.iloc[-2]
    sl_buffer = 10 * pip_size

    is_ha_green = last_closed_ha_candle['close'] > last_closed_ha_candle['open']
    # A strong buy signal candle has no lower wick.
    is_strong_buy_candle = last_closed_ha_candle['open'] == last_closed_ha_candle['low']
    was_prev_ha_red = prev_ha_candle['close'] < prev_ha_candle['open']

    reason = "Pure Heikin Ashi Reversal"

    if is_ha_green and is_strong_buy_candle and was_prev_ha_red:
        # Use a unique key for pure HA signals to avoid conflicts with MA signals
        if last_signal_timestamps.get(f"{symbol}_pure_ha") != last_closed_ha_candle['time']:
            last_signal_timestamps[f"{symbol}_pure_ha"] = last_closed_ha_candle['time']
            signal_price = last_closed_raw_candle['close']
            # Place SL below the low of the signal candle
            sl_price = last_closed_raw_candle['low'] - sl_buffer
            risk = signal_price - sl_price
            tp_price = signal_price + (risk * RISK_REWARD_RATIO)

            signal_info = {
                'type': 'BUY', 'symbol': symbol, 'price': signal_price,
                'sl': sl_price, 'tp': tp_price, 'time': last_closed_ha_candle['time'],
                'reason': reason,
                'mode': 'pure_ha'  # Identify signal mode
            }
            # Simplified exits for this mode
            exits = {'conservative_exit': "Close if Heikin Ashi candle color flips."}
            signal_info['exits'] = exits
            return signal_info
    return None


def check_sell_signal_pure_ha(symbol, ha_df, rates_df, pip_size):
    """Checks for a sell signal based on pure Heikin Ashi reversal."""
    if len(ha_df) < 3:
        return None

    prev_ha_candle = ha_df.iloc[-3]
    last_closed_ha_candle = ha_df.iloc[-2]
    last_closed_raw_candle = rates_df.iloc[-2]
    sl_buffer = 10 * pip_size

    is_ha_red = last_closed_ha_candle['close'] < last_closed_ha_candle['open']
    # A strong sell signal candle has no upper wick.
    is_strong_sell_candle = last_closed_ha_candle['open'] == last_closed_ha_candle['high']
    was_prev_ha_green = prev_ha_candle['close'] > prev_ha_candle['open']

    reason = "Pure Heikin Ashi Reversal"

    if is_ha_red and is_strong_sell_candle and was_prev_ha_green:
        # Use a unique key for pure HA signals to avoid conflicts with MA signals
        if last_signal_timestamps.get(f"{symbol}_pure_ha") != last_closed_ha_candle['time']:
            last_signal_timestamps[f"{symbol}_pure_ha"] = last_closed_ha_candle['time']
            signal_price = last_closed_raw_candle['close']
            # Place SL above the high of the signal candle
            sl_price = last_closed_raw_candle['high'] + sl_buffer
            risk = sl_price - signal_price
            tp_price = signal_price - (risk * RISK_REWARD_RATIO)

            signal_info = {
                'type': 'SELL', 'symbol': symbol, 'price': signal_price,
                'sl': sl_price, 'tp': tp_price, 'time': last_closed_ha_candle['time'],
                'reason': reason,
                'mode': 'pure_ha'  # Identify signal mode
            }
            # Simplified exits for this mode
            exits = {'conservative_exit': "Close if Heikin Ashi candle color flips."}
            signal_info['exits'] = exits
            return signal_info
    return None


def check_and_get_signal(symbol):
    """
    Performs the core signal analysis for a symbol.
    It checks for all available signal types (MA cross, pure HA) and returns a
    list of all valid signals found.
    """
    signals = []
    # Increased count to have enough data for pattern analysis
    rates_df = get_price_data(symbol, TIMEFRAME, count=200)
    if rates_df is None or rates_df.empty or len(rates_df) < SLOW_MA_PERIOD + 50:  # Increased buffer
        logger.warning(f"Not enough data to analyze {symbol} for a signal.")
        return signals

    # --- Common Data Preparation ---
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.warning(f"Could not get info for {symbol}")
        return signals
    pip_size = symbol_info.point

    # Calculate Heikin Ashi candles from the original rates_df
    ha_df = calculate_heikin_ashi(rates_df.copy())

    # --- Mode 1: HA + MA Crossover Analysis ---
    # Calculate indicators on the raw price data first
    rates_df_with_indicators = calculate_indicators(rates_df.copy())
    # Add the indicators to the Heikin Ashi dataframe for checking
    ha_df_with_ma = ha_df.copy()
    ha_df_with_ma['fast_ma'] = rates_df_with_indicators['fast_ma']
    ha_df_with_ma['slow_ma'] = rates_df_with_indicators['slow_ma']

    buy_signal_ha_ma = check_buy_signal(symbol, ha_df_with_ma, rates_df, pip_size)
    if buy_signal_ha_ma:
        signals.append(buy_signal_ha_ma)

    sell_signal_ha_ma = check_sell_signal(symbol, ha_df_with_ma, rates_df, pip_size)
    if sell_signal_ha_ma:
        signals.append(sell_signal_ha_ma)

    # --- Mode 2: Pure Heikin Ashi Analysis ---
    buy_signal_pure_ha = check_buy_signal_pure_ha(symbol, ha_df, rates_df, pip_size)
    if buy_signal_pure_ha:
        signals.append(buy_signal_pure_ha)

    sell_signal_pure_ha = check_sell_signal_pure_ha(symbol, ha_df, rates_df, pip_size)
    if sell_signal_pure_ha:
        signals.append(sell_signal_pure_ha)

    return signals


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


def get_signal_for_symbol(symbol: str, chat_id: int) -> tuple[str, dict | None]:
    """
    Wrapper function for the /signal command.
    Analyzes a single symbol, filters for the user's mode, and returns a signal
    message or a 'no signal' message.
    """
    # Ensure MT5 is connected before proceeding
    if not mt5.terminal_info().connected:
        if not initialize_mt5():
            return "Error: Could not connect to the trading server. Please try again later.", None

    all_signals = check_and_get_signal(symbol)
    user_mode = get_user_settings(chat_id).get('mode', 'ha_ma')

    # Filter signals based on the user's chosen mode
    filtered_signal = None
    for signal in all_signals:
        if signal.get('mode') == user_mode:
            filtered_signal = signal
            break  # Take the first matching signal for this user

    if filtered_signal:
        # Return both the formatted message and the raw signal data
        return format_signal_message(filtered_signal), filtered_signal
    else:
        # Check for data availability to give a more informative message
        rates_df = get_price_data(symbol, TIMEFRAME, count=2)
        if rates_df is None or rates_df.empty:
            return f"Could not retrieve data for {symbol}. It might be an invalid symbol.", None
        return f"No clear signal for {symbol} using your selected mode at the moment.", None


def place_market_order(signal_type: str, symbol: str, lot_size: float, sl_price: float, tp_price: float, mode: str, chat_id: int) -> str:
    """
    Places a market order on MetaTrader 5.
    Returns a string with the result of the operation.
    """
    logger.info(f"Attempting to place {signal_type} order for {symbol} with lot size {lot_size} for chat_id {chat_id}.")

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

    # If mode is ha_ma, TP is dynamic, so set it to 0.0 to disable it.
    if mode == 'ha_ma':
        tp_price = 0.0
        logger.info(f"Order mode is 'ha_ma'. Setting TP to 0.0 for dynamic management.")

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
        "comment": f"Telegram Signal Bot - {mode}",
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
            # Store the mapping of ticket to chat_id for future notifications
            trade_to_chat_id[result.order] = chat_id
            logger.info(f"Stored mapping for ticket {result.order} to chat_id {chat_id}")
            return (f"✅ **Order Successful!**\n\n"
                    f"🔹 **Symbol:** {symbol}\n"
                    f"🔹 **Type:** {signal_type}\n"
                    f"🔹 **Lot Size:** {result.volume}\n"
                    f"🔹 **Ticket:** {result.order}")
    except Exception as e:
        logger.error(f"An exception occurred while placing order for {symbol}: {e}")
        return f"❌ **Order Failed**: An unexpected error occurred: {e}"


def close_position(position, reason: str) -> bool:
    """
    Closes an open position on MetaTrader 5.
    """
    symbol = position.symbol
    volume = position.volume
    ticket = position.ticket
    order_type = position.type

    logger.info(f"Attempting to close position #{ticket} for {symbol} ({volume} lots). Reason: {reason}")

    # Determine the closing order type and price
    if order_type == mt5.ORDER_TYPE_BUY:
        close_order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    elif order_type == mt5.ORDER_TYPE_SELL:
        close_order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    else:
        logger.error(f"Unknown order type {order_type} for ticket {ticket}.")
        return False

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_order_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": reason,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    try:
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position {ticket}. Code: {result.retcode}, Comment: {result.comment}")
            return False
        else:
            logger.info(f"Successfully closed position {ticket}. Result: {result}")
            # Remove the trade from our tracking dictionary
            if ticket in trade_to_chat_id:
                del trade_to_chat_id[ticket]
                logger.info(f"Removed ticket {ticket} from tracking dictionary.")
            return True
    except Exception as e:
        logger.error(f"An exception occurred while closing position {ticket}: {e}")
        return False


# --- Telegram Command Handlers ---

async def start(update: Update, context) -> None:
    """Sends a welcome message with a main menu."""
    user = update.effective_user
    welcome_message = f"👋 Hello {user.first_name}!\n\nI am your Forex Signal & Trading Bot. Please choose an option below to begin."

    keyboard = [
        [
            InlineKeyboardButton("📊 Get Signal", callback_data='menu_signal'),
            InlineKeyboardButton("📈 Account Status", callback_data='menu_status')
        ],
        [
            InlineKeyboardButton("⚙️ Monitoring", callback_data='menu_monitor'),
            InlineKeyboardButton("📉 PnL Check", callback_data='menu_pnl')
        ],
        [
            InlineKeyboardButton("🤖 Auto-Trade", callback_data='menu_autotrade'),
            InlineKeyboardButton("🔧 Signal Mode", callback_data='menu_mode')
        ],
        [InlineKeyboardButton("❓ Help", callback_data='menu_help')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # If the command was initiated by a button press (e.g., from help menu), edit the message.
    if update.callback_query:
        await update.callback_query.edit_message_text(
            text=welcome_message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            text=welcome_message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

async def help_command(update: Update, context) -> None:
    """Sends a detailed help message with all commands."""
    help_message = (
        "Here is a list of available commands and features:\n\n"
        "**Signal Commands**\n"
        "🔹 /signal - Get an immediate signal for a currency pair.\n"
        "🔹 /monitor - Select pairs to get continuous signal alerts for.\n"
        "🔹 /unmonitor - Stop monitoring specific pairs.\n"
        "🔹 /monitoring - See the list of pairs you are monitoring.\n\n"
        "**Trading & Account Commands**\n"
        "📈 /status - Check your account balance, equity, and PnL.\n"
        "📉 /pnl - Check the PnL for a specific open order.\n"
        "🤖 /autotrade_on - Enable automatic trade execution (0.01 lot).\n"
        "🤖 /autotrade_off - Disable automatic trade execution.\n"
        "🤖 /autotrade_status - Check if auto-trading is on or off.\n\n"
        "**Settings**\n"
        "⚙️ /set_mode - Choose between different signal strategies.\n"
        "ℹ️ /mode_status - Check your current signal strategy.\n\n"
        "You can always return to the main menu by sending /start."
    )
    keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data='menu_start')]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await update.callback_query.edit_message_text(text=help_message, reply_markup=reply_markup, parse_mode='Markdown')
    else:
        await update.message.reply_text(text=help_message, reply_markup=reply_markup, parse_mode='Markdown')


async def autotrade_menu(update: Update, context) -> None:
    """Shows autotrade status and toggle buttons."""
    chat_id = update.effective_chat.id
    settings = get_user_settings(chat_id)
    status = "ENABLED" if settings.get('auto_trade', False) else "DISABLED"

    message = f"🤖 Auto-trading is currently **{status}**.\n\n" \
              "When enabled, I will execute trades automatically with a lot size of 0.01. " \
              "When disabled, I will ask for confirmation."

    keyboard = [
        [
            InlineKeyboardButton("✅ Enable", callback_data='autotrade_on'),
            InlineKeyboardButton("❌ Disable", callback_data='autotrade_off')
        ],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data='menu_start')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # This function is triggered by a button press, so we edit the message.
    await update.callback_query.edit_message_text(
        text=message,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )


async def set_mode(update: Update, context) -> None:
    """Displays an inline keyboard for the user to select a signal mode. Handles both commands and callbacks."""
    query = update.callback_query
    if query:
        await query.answer()

    keyboard = [
        [InlineKeyboardButton("Mode 1: HA + MA Crossover", callback_data='mode_ha_ma')],
        [InlineKeyboardButton("Mode 2: Pure Heikin Ashi", callback_data='mode_pure_ha')],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data='menu_start')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    text = 'Please choose your preferred signal mode:'

    if query:
        await query.edit_message_text(text=text, reply_markup=reply_markup)
    else:
        await update.message.reply_text(text=text, reply_markup=reply_markup)


async def mode_status(update: Update, context) -> None:
    """Checks the current signal mode for the user."""
    chat_id = update.message.chat_id
    settings = get_user_settings(chat_id)
    mode = settings.get('mode', 'ha_ma')  # Default for safety
    if mode == 'ha_ma':
        mode_description = "Mode 1: HA + MA Crossover"
    else:
        mode_description = "Mode 2: Pure Heikin Ashi"
    await update.message.reply_text(f"Your current signal mode is set to:\n**{mode_description}**", parse_mode='Markdown')


async def autotrade_on(update: Update, context) -> None:
    """Enables auto-trading for the user. Handles both commands and callbacks."""
    chat_id = update.effective_chat.id
    settings = get_user_settings(chat_id)
    settings['auto_trade'] = True
    logger.info(f"Auto-trading enabled for chat_id: {chat_id}")
    message_text = "🤖 Auto-trading has been **ENABLED**. I will now execute trades automatically with a lot size of 0.01."

    if update.callback_query:
        # User came from a button press, so edit the message.
        await update.callback_query.edit_message_text(text=message_text, parse_mode='Markdown')
    else:
        # User sent a command, so reply.
        await update.message.reply_text(text=message_text, parse_mode='Markdown')


async def autotrade_off(update: Update, context) -> None:
    """Disables auto-trading for the user. Handles both commands and callbacks."""
    chat_id = update.effective_chat.id
    settings = get_user_settings(chat_id)
    settings['auto_trade'] = False
    logger.info(f"Auto-trading disabled for chat_id: {chat_id}")
    message_text = "🤖 Auto-trading has been **DISABLED**. I will ask for confirmation before placing trades."

    if update.callback_query:
        # User came from a button press, so edit the message.
        await update.callback_query.edit_message_text(text=message_text, parse_mode='Markdown')
    else:
        # User sent a command, so reply.
        await update.message.reply_text(text=message_text, parse_mode='Markdown')


async def autotrade_status(update: Update, context) -> None:
    """Checks the current auto-trading status for the user."""
    chat_id = update.message.chat_id
    settings = get_user_settings(chat_id)
    status = settings.get('auto_trade', False)
    if status:
        await update.message.reply_text("🤖 Auto-trading is currently **ENABLED**.")
    else:
        await update.message.reply_text("🤖 Auto-trading is currently **DISABLED**.")


async def account_status(update: Update, context) -> None:
    """Displays the current MetaTrader account status. Works with commands and callbacks."""
    message = update.effective_message
    if not mt5.terminal_info().connected:
        if not initialize_mt5():
            await message.reply_text("❌ Could not connect to the trading server. Please try again later.")
            return

    account_info = mt5.account_info()
    if not account_info:
        await message.reply_text("❌ Failed to retrieve account information.")
        return

    positions = mt5.positions_get()
    if positions is None:
        await message.reply_text("❌ Failed to retrieve open positions. The connection might have been lost.")
        return

    num_positions = len(positions)
    total_pnl = sum(pos.profit for pos in positions)

    pnl_icon = "🟢" if total_pnl >= 0 else "🔴"
    status_message = (
        f"📊 **Account Status**\n\n"
        f"🔹 **Balance:** {account_info.balance:.2f} {account_info.currency}\n"
        f"🔹 **Equity:** {account_info.equity:.2f} {account_info.currency}\n"
        f"🔹 **Open Positions:** {num_positions}\n"
        f"{pnl_icon} **Total PnL:** {total_pnl:.2f} {account_info.currency}"
    )

    await message.reply_text(status_message, parse_mode='Markdown')


async def pnl_command(update: Update, context) -> None:
    """Starts the conversation to check PnL for a specific order. Handles both commands and callbacks."""
    message = update.effective_message
    query = update.callback_query

    if query:
        await query.answer()

    # Ensure MT5 connection
    if not mt5.terminal_info().connected:
        if not initialize_mt5():
            text = "❌ Could not connect to the trading server. Please try again later."
            if query: await query.edit_message_text(text)
            else: await message.reply_text(text)
            return

    positions = mt5.positions_get()
    if positions is None:
        text = "❌ Failed to retrieve open positions. The connection might have been lost."
        if query: await query.edit_message_text(text)
        else: await message.reply_text(text)
        return

    if not positions:
        text = "ℹ️ You have no open orders to check."
        if query: await query.edit_message_text(text)
        else: await message.reply_text(text)
        return

    keyboard = []
    for pos in positions:
        trade_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
        button_text = f"{trade_type} {pos.symbol} {pos.volume} lot"
        callback_data = f"pnl_{pos.ticket}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

    # Add a back button if called from a menu for better navigation
    if query:
        keyboard.append([InlineKeyboardButton("⬅️ Back to Main Menu", callback_data='menu_start')])


    reply_markup = InlineKeyboardMarkup(keyboard)
    text = "Please select an order to check its PnL:"
    if query:
        await query.edit_message_text(text=text, reply_markup=reply_markup)
    else:
        await message.reply_text(text=text, reply_markup=reply_markup)


async def pnl_callback(update: Update, context) -> None:
    """Handles the PnL button press and shows details for a specific order."""
    query = update.callback_query
    await query.answer()

    try:
        _, ticket_str = query.data.split('_')
        ticket = int(ticket_str)
    except (ValueError, IndexError):
        await query.edit_message_text("❌ Invalid callback data.")
        return

    if not mt5.terminal_info().connected:
        if not initialize_mt5():
            await query.edit_message_text("❌ Could not connect to the trading server.")
            return

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        await query.edit_message_text(f"❌ Could not find order with ticket {ticket}. It might have been closed.")
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

    await query.edit_message_text(text=message, parse_mode='Markdown')


async def trade_callback(update: Update, context) -> int:
    """Handles the 'BUY' or 'SELL' button press to start the trade conversation."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id

    # e.g., 'trade_buy_EURUSD'
    _, trade_type, symbol = query.data.split('_')

    # It's crucial to fetch the latest signal data right before the trade
    # to ensure prices (SL/TP) are as relevant as possible.
    all_signals = check_and_get_signal(symbol)
    user_mode = get_user_settings(chat_id).get('mode', 'ha_ma')

    # Find a signal that matches the user's mode and the action they clicked
    signal_to_trade = None
    for s in all_signals:
        if s['mode'] == user_mode and s['type'].lower() == trade_type:
            signal_to_trade = s
            break

    if not signal_to_trade:
        await query.edit_message_text(text="⚠️ The signal has expired or conditions have changed. Please request a new signal analysis.")
        return ConversationHandler.END

    # Store the validated signal data in user_data for the next step
    context.user_data['trade_signal'] = signal_to_trade
    logger.info(f"User {query.from_user.id} initiated {trade_type} for {symbol}. Asking for lot size.")

    # Edit the original message to ask for lot size
    await query.edit_message_text(
        text=f"You've initiated a **{trade_type.upper()}** for **{symbol}**.\n\n"
             f"Please enter the lot size you wish to use (e.g., `0.01`, `0.1`).\n\n"
             f"Type /cancel to abort."
    )

    return ASKING_LOT_SIZE


async def lot_size_input(update: Update, context) -> int:
    """Receives lot size, places the order, and ends the conversation."""
    lot_size_str = update.message.text
    signal = context.user_data.get('trade_signal')

    if not signal:
        await update.message.reply_text("❌ An error occurred: I've lost the context of the trade. Please start over by requesting a new signal.")
        return ConversationHandler.END

    try:
        # Validate the lot size
        lot_size = float(lot_size_str)
        if lot_size <= 0:
            raise ValueError("Lot size must be a positive number.")
    except ValueError:
        await update.message.reply_text(
            "⚠️ Invalid lot size. Please enter a positive number (e.g., `0.01`).\n\n"
            "Or type /cancel to abort."
        )
        return ASKING_LOT_SIZE # Ask again without ending the conversation

    logger.info(f"User {update.effective_user.id} entered lot size {lot_size}. Placing order.")
    await update.message.reply_text(f"⏳ Understood. Placing a **{signal['type']}** order for **{signal['symbol']}** with lot size **{lot_size}**...")

    # Execute the trade
    trade_result = place_market_order(
        signal_type=signal['type'],
        symbol=signal['symbol'],
        lot_size=lot_size,
        sl_price=signal['sl'],
        tp_price=signal['tp'],
        mode=signal['mode'],
        chat_id=update.message.chat_id
    )

    await update.message.reply_text(trade_result, parse_mode='Markdown')

    # Clean up the user_data to free memory
    if 'trade_signal' in context.user_data:
        del context.user_data['trade_signal']

    return ConversationHandler.END


async def signal_command(update: Update, context) -> int:
    """Displays an inline keyboard for the user to select a symbol for a one-time signal. Starts the conversation."""
    message = update.effective_message
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

    text = 'Please choose a symbol to get a signal for, or choose "Custom":'
    if update.callback_query:
        await message.edit_text(text, reply_markup=reply_markup)
    else:
        await message.reply_text(text, reply_markup=reply_markup)

    return CHOOSING_SIGNAL


async def signal_button_callback(update: Update, context) -> int:
    """Handles button presses for the signal command, gets signal, and shows trade buttons."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id

    action, *data = query.data.split('_', 1)
    symbol = data[0] if data else None

    if symbol == 'custom':
        await query.edit_message_text(text="Please send me the symbol you want to analyze (e.g., EURUSD, BTCUSD).")
        return TYPING_CUSTOM_SIGNAL

    if symbol:
        await query.edit_message_text(text=f"⏳ Analyzing {symbol}, please wait...")
        signal_message, signal_data = get_signal_for_symbol(symbol, chat_id)

        if signal_data:
            # If there's a signal, show only the correct trade button
            signal_type = signal_data['type']
            if signal_type == 'BUY':
                button = InlineKeyboardButton(f"📈 BUY {symbol}", callback_data=f"trade_buy_{symbol}")
            else:  # SELL
                button = InlineKeyboardButton(f"📉 SELL {symbol}", callback_data=f"trade_sell_{symbol}")

            keyboard = [[button]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text=signal_message, reply_markup=reply_markup)
        else:
            # If no signal, just show the message
            await query.edit_message_text(text=signal_message)

    return ConversationHandler.END


async def custom_signal_input(update: Update, context) -> int:
    """Handles custom symbol input, gets signal, and shows trade buttons."""
    symbol = update.message.text.upper()
    chat_id = update.message.chat_id

    await update.message.reply_text(f"⏳ Analyzing {symbol}, please wait...")
    signal_message, signal_data = get_signal_for_symbol(symbol, chat_id)

    if signal_data:
        # If there's a signal, show only the correct trade button
        signal_type = signal_data['type']
        # Get symbol from signal_data to ensure consistency
        symbol = signal_data['symbol']
        if signal_type == 'BUY':
            button = InlineKeyboardButton(f"📈 BUY {symbol}", callback_data=f"trade_buy_{symbol}")
        else:  # SELL
            button = InlineKeyboardButton(f"📉 SELL {symbol}", callback_data=f"trade_sell_{symbol}")

        keyboard = [[button]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(text=signal_message, reply_markup=reply_markup)
    else:
        # If no signal, just show the message
        await update.message.reply_text(text=signal_message)

    return ConversationHandler.END


async def cancel_conversation(update: Update, context) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text('Operation cancelled.')
    return ConversationHandler.END


async def monitor_command(update: Update, context) -> None:
    """
    Adds a symbol to the monitoring list via text command,
    or displays an inline keyboard for the user to select symbols to monitor.
    """
    message = update.effective_message
    chat_id = message.chat_id

    # Check if the command was used with an argument (e.g., /monitor BTCUSD)
    if context.args:
        symbol = context.args[0].upper()
        # Initialize list if not present
        if chat_id not in monitored_pairs:
            monitored_pairs[chat_id] = []

        # Add symbol if not already there
        if symbol not in monitored_pairs[chat_id]:
            monitored_pairs[chat_id].append(symbol)
            logger.info(f"Added {symbol} to monitor list for chat_id {chat_id}")
            await message.reply_text(
                f"✅ Added **{symbol}** to your monitoring list.\n"
                f"I will check for signals every 5 minutes."
            , parse_mode='Markdown')
        else:
            await message.reply_text(f"ℹ️ You are already monitoring {symbol}.")
        return

    # If no argument, show the buttons
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

    text = 'Choose symbols to add to your monitoring list, or use `/monitor SYMBOL`:'
    if update.callback_query:
        await message.edit_text(text, reply_markup=reply_markup)
    else:
        await message.reply_text(text, reply_markup=reply_markup)


async def unmonitor_command(update: Update, context) -> None:
    """Displays an inline keyboard of currently monitored symbols for the user to remove."""
    message = update.effective_message
    chat_id = message.chat_id
    user_monitored_pairs = monitored_pairs.get(chat_id, [])

    if not user_monitored_pairs:
        await message.reply_text("You are not monitoring any pairs yet. Use /monitor to add some.")
        return

    keyboard = [[InlineKeyboardButton(f"❌ {symbol}", callback_data=f'unmonitor_{symbol}')] for symbol in user_monitored_pairs]
    keyboard.append([
        InlineKeyboardButton("🗑️ Clear All", callback_data='unmonitor_clear_all'),
        InlineKeyboardButton("✅ Done", callback_data='unmonitor_done')
    ])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await message.reply_text('Choose symbols to remove from your monitoring list:', reply_markup=reply_markup)


async def monitoring_command(update: Update, context) -> None:
    """Lists the symbols the user is currently monitoring."""
    chat_id = update.message.chat_id
    user_monitored_pairs = monitored_pairs.get(chat_id, [])

    if not user_monitored_pairs:
        message = "You are not monitoring any pairs yet. Use /monitor to add some."
    else:
        message = "You are currently monitoring the following pairs:\n" + "\n".join(f"• {s}" for s in user_monitored_pairs)

    await update.message.reply_text(message)


async def button_callback(update: Update, context) -> None:
    """Parses the CallbackQuery and runs the appropriate action for monitoring."""
    query = update.callback_query
    await query.answer()

    action, *data = query.data.split('_', 1)
    symbol = data[0] if data else None
    chat_id = query.message.chat_id

    # --- Monitor Action ---
    if action == 'monitor':
        if symbol == 'done':
            user_monitored_pairs = monitored_pairs.get(chat_id, [])
            if user_monitored_pairs:
                final_message = "✅ Your monitoring list is updated. I will now watch:\n" + "\n".join(f"• {s}" for s in user_monitored_pairs)
            else:
                final_message = "Your monitoring list is empty. Add pairs to start receiving signals."
            await query.edit_message_text(text=final_message)
            return

        # Initialize list if not present
        if chat_id not in monitored_pairs:
            monitored_pairs[chat_id] = []

        # Add symbol if not already there
        if symbol and symbol not in monitored_pairs[chat_id]:
            monitored_pairs[chat_id].append(symbol)
            await query.answer(text=f"✅ Added {symbol}")
        elif symbol:
            await query.answer(text=f"ℹ️ Already monitoring {symbol}")

        # --- Refresh the message with updated list ---
        user_monitored_pairs = monitored_pairs.get(chat_id, [])
        monitored_list_text = "\n\n**Currently Monitoring:**\n" + "\n".join(f"• {s}" for s in user_monitored_pairs) if user_monitored_pairs else ""

        # Re-create the keyboard to show the same options
        keyboard = [
            [InlineKeyboardButton("EURUSD", callback_data='monitor_EURUSD'), InlineKeyboardButton("GBPUSD", callback_data='monitor_GBPUSD')],
            [InlineKeyboardButton("USDJPY", callback_data='monitor_USDJPY'), InlineKeyboardButton("AUDUSD", callback_data='monitor_AUDUSD')],
            [InlineKeyboardButton("USDCAD", callback_data='monitor_USDCAD'), InlineKeyboardButton("XAUUSD", callback_data='monitor_XAUUSD')],
            [InlineKeyboardButton("✅ Done", callback_data='monitor_done')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            text=f'Choose symbols to add to your monitoring list.{monitored_list_text}',
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    # --- Unmonitor Action ---
    elif action == 'unmonitor':
        if symbol == 'done':
            await query.edit_message_text(text="Your monitoring list has been updated.")
            return

        if symbol == 'clear_all':
            if chat_id in monitored_pairs:
                monitored_pairs[chat_id] = []
                logger.info(f"Cleared all monitored pairs for chat_id {chat_id}")
            await query.edit_message_text("✅ All symbols have been removed from your monitoring list.")
            return

        # --- Remove a single symbol ---
        if chat_id in monitored_pairs and symbol in monitored_pairs[chat_id]:
            monitored_pairs[chat_id].remove(symbol)
            await query.answer(text=f"❌ Removed {symbol}")

            # Refresh the keyboard to show the updated list
            user_monitored_pairs = monitored_pairs.get(chat_id, [])
            if not user_monitored_pairs:
                await query.edit_message_text("You are no longer monitoring any pairs.")
            else:
                keyboard = [[InlineKeyboardButton(f"❌ {s}", callback_data=f'unmonitor_{s}')] for s in user_monitored_pairs]
                keyboard.append([
                    InlineKeyboardButton("🗑️ Clear All", callback_data='unmonitor_clear_all'),
                    InlineKeyboardButton("✅ Done", callback_data='unmonitor_done')
                ])
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text('Choose symbols to remove:', reply_markup=reply_markup)
        else:
            await query.answer(text=f"ℹ️ You were not monitoring {symbol}.")

    else:
        await query.edit_message_text(text=f"Unknown action: {action}")


async def mode_callback(update: Update, context) -> None:
    """Handles the mode selection button press."""
    query = update.callback_query
    await query.answer()

    # e.g., 'mode_ha_ma' or 'mode_pure_ha'
    try:
        action, mode = query.data.split('_', 1)
        if action != 'mode':
            # Should not happen if pattern is correct, but good practice
            return
    except (ValueError, IndexError):
        await query.edit_message_text("❌ Invalid callback data for mode selection.")
        return

    chat_id = query.message.chat_id
    settings = get_user_settings(chat_id)
    # The mode from callback will be 'ha_ma' or 'pure_ha'
    settings['mode'] = mode

    logger.info(f"User {chat_id} set signal mode to {mode}")

    if mode == 'ha_ma':
        mode_description = "Mode 1: HA + MA Crossover"
    else:
        mode_description = "Mode 2: Pure Heikin Ashi"

    await query.edit_message_text(
        f"✅ Your signal mode has been updated to:\n**{mode_description}**",
        parse_mode='Markdown'
    )


async def check_all_monitored_pairs_job(context) -> None:
    """
    This job first manages open positions, then iterates through all monitored
    pairs to check for new signals.
    """
    # First, manage existing positions based on dynamic TP rules
    await check_and_manage_open_positions(context)

    if not mt5.terminal_info().connected:
        logger.warning("Job running, but MT5 is not connected. Attempting to reconnect.")
        if not initialize_mt5():
            logger.error("Job failed to reconnect to MT5. Skipping this run.")
            return

    logger.info("Running scheduled job: Checking for new signals...")

    # Create a copy of the items to avoid issues if the dict is modified during iteration
    # Get a unique list of all symbols being monitored to avoid re-checking the same symbol
    all_symbols_to_check = set()
    for symbols in monitored_pairs.values():
        all_symbols_to_check.update(symbols)

    # Check each symbol once and store the results
    signals_by_symbol = {symbol: check_and_get_signal(symbol) for symbol in all_symbols_to_check}

    for chat_id, symbols in list(monitored_pairs.items()):
        user_mode = get_user_settings(chat_id).get('mode', 'ha_ma')
        for symbol in symbols:
            try:
                all_signals = signals_by_symbol.get(symbol, [])
                if not all_signals:
                    continue

                # Filter signals for the current user's mode
                for signal in all_signals:
                    if signal.get('mode') == user_mode:
                        # Found a matching signal, process it
                        is_auto_trade = get_user_settings(chat_id).get('auto_trade', False)

                        if is_auto_trade:
                            logger.info(f"Auto-trading is ON for chat_id {chat_id}. Placing trade for {symbol}.")
                            trade_result = place_market_order(
                                signal_type=signal['type'],
                                symbol=signal['symbol'],
                                lot_size=0.01,
                                sl_price=signal['sl'],
                                tp_price=signal['tp'],
                                mode=signal['mode'],
                                chat_id=chat_id
                            )
                            auto_trade_message = f"🤖 **Auto-Trade Executed** for {symbol}.\n\n{trade_result}"
                            await context.bot.send_message(chat_id=chat_id, text=auto_trade_message, parse_mode='Markdown')
                        else:
                            # Manual trade: send signal with the correct action button
                            message = format_signal_message(signal)
                            signal_type = signal['type']
                            if signal_type == 'BUY':
                                button = InlineKeyboardButton(f"📈 BUY {symbol}", callback_data=f"trade_buy_{symbol}")
                            else:  # SELL
                                button = InlineKeyboardButton(f"📉 SELL {symbol}", callback_data=f"trade_sell_{symbol}")

                            keyboard = [[button]]
                            reply_markup = InlineKeyboardMarkup(keyboard)
                            await context.bot.send_message(chat_id=chat_id, text=message, reply_markup=reply_markup)

                        logger.info(f"Sent signal for {symbol} to chat_id {chat_id} (Mode: {user_mode}, Auto-Trade: {is_auto_trade})")
                        # Break after processing the first valid signal for this symbol/user combination
                        break

            except Exception as e:
                logger.error(f"Error checking signal for {symbol} for chat_id {chat_id}: {e}")


async def check_and_manage_open_positions(context) -> None:
    """
    Checks all open positions and closes them if the reverse MA-crossover signal appears.
    This only applies to trades opened with the 'ha_ma' strategy.
    """
    try:
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            return  # No open positions to manage

        logger.info(f"Managing {len(positions)} open position(s)...")

        # Iterate over a copy, as the underlying list of positions can change.
        for position in list(positions):
            # Filter for trades managed by this bot and by the dynamic TP strategy
            if position.magic != 234000 or "ha_ma" not in position.comment:
                continue

            symbol = position.symbol
            logger.info(f"Checking dynamic TP for position #{position.ticket} on {symbol}.")

            # Get fresh data to check for crossover
            rates_df = get_price_data(symbol, TIMEFRAME, count=SLOW_MA_PERIOD + 5)
            if rates_df is None or len(rates_df) < SLOW_MA_PERIOD + 3:
                logger.warning(f"Not enough data to manage position #{position.ticket} on {symbol}.")
                continue

            rates_df = calculate_indicators(rates_df)

            # We need at least two candles with MAs to check for a cross
            if rates_df['slow_ma'].isna().sum() > len(rates_df) - 3:
                logger.warning(f"Not enough MA data to manage position #{position.ticket} on {symbol}.")
                continue

            last_candle = rates_df.iloc[-2]
            prev_candle = rates_df.iloc[-3]

            close_condition_met = False
            reason = ""

            # Check for bearish cross to close a BUY position
            if position.type == mt5.ORDER_TYPE_BUY:
                is_bearish_cross = prev_candle['fast_ma'] >= prev_candle['slow_ma'] and last_candle['fast_ma'] < last_candle['slow_ma']
                if is_bearish_cross:
                    close_condition_met = True
                    reason = "Dynamic TP hit (Bearish MA Crossover)"

            # Check for bullish cross to close a SELL position
            elif position.type == mt5.ORDER_TYPE_SELL:
                is_bullish_cross = prev_candle['fast_ma'] <= prev_candle['slow_ma'] and last_candle['fast_ma'] > last_candle['slow_ma']
                if is_bullish_cross:
                    close_condition_met = True
                    reason = "Dynamic TP hit (Bullish MA Crossover)"

            if close_condition_met:
                logger.info(f"Closing condition met for position #{position.ticket}. Reason: {reason}")

                trade_type_str = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                chat_id = trade_to_chat_id.get(position.ticket)

                if close_position(position, reason):
                    logger.info(f"Successfully closed position #{position.ticket} via dynamic TP.")
                    if chat_id:
                        message = (
                            f"✅ **Position Closed Automatically**\n\n"
                            f"🔹 **Symbol:** {symbol}\n"
                            f"🔹 **Type:** {trade_type_str}\n"
                            f"🔹 **Ticket:** {position.ticket}\n"
                            f"🔹 **Reason:** {reason}"
                        )
                        try:
                            await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                        except Exception as e:
                            logger.error(f"Failed to send closure notification to chat_id {chat_id} for ticket {position.ticket}: {e}")
                    else:
                        logger.warning(f"Could not find chat_id for closed ticket {position.ticket} to send notification.")
                else:
                    logger.error(f"Failed to execute closure for position #{position.ticket} despite condition being met.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in check_and_manage_open_positions: {e}")


# --- Main Bot Logic ---
async def main_menu_handler(update: Update, context) -> None:
    """Handles callbacks from the main menu."""
    query = update.callback_query
    await query.answer()

    if query.data == 'menu_help':
        await help_command(update, context)
    elif query.data == 'menu_status':
        await account_status(update, context)
    elif query.data == 'menu_monitor':
        await monitor_command(update, context)
    elif query.data == 'menu_start':
        await start(update, context)
    elif query.data == 'menu_pnl':
        await pnl_command(update, context)
    elif query.data == 'menu_autotrade':
        await autotrade_menu(update, context)
    elif query.data == 'menu_mode':
        await set_mode(update, context)
    # The 'menu_signal' is handled by the ConversationHandler entry point now


def main() -> None:
    """Start the telegram bot."""
    # Initialize MT5
    if not initialize_mt5():
        logger.error("Failed to initialize MetaTrader 5. The bot will not be able to fetch data.")
        # Decide if you want the bot to run without MT5 connection.
        # For this example, we'll let it run to allow user interaction,
        # but signal-related commands will fail.

    # Create the Application and pass it your bot's token.
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("Telegram Bot Token is not configured. Please set the TELEGRAM_BOT_TOKEN environment variable.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("monitor", monitor_command))
    application.add_handler(CommandHandler("unmonitor", unmonitor_command))
    application.add_handler(CommandHandler("monitoring", monitoring_command))
    application.add_handler(CommandHandler("set_mode", set_mode))
    application.add_handler(CommandHandler("mode_status", mode_status))
    application.add_handler(CommandHandler("autotrade_on", autotrade_on))
    application.add_handler(CommandHandler("autotrade_off", autotrade_off))
    application.add_handler(CommandHandler("autotrade_status", autotrade_status))
    application.add_handler(CommandHandler("status", account_status))
    application.add_handler(CommandHandler("pnl", pnl_command))

    # Set up the conversation handler for the /signal command
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('signal', signal_command),
            CallbackQueryHandler(signal_command, pattern='^menu_signal$')
        ],
        states={
            CHOOSING_SIGNAL: [CallbackQueryHandler(signal_button_callback, pattern='^signal_')],
            TYPING_CUSTOM_SIGNAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_signal_input)],
        },
        fallbacks=[CommandHandler('cancel', cancel_conversation)],
        per_message=False
    )
    application.add_handler(conv_handler)

    # This handler processes button clicks from the main menu
    application.add_handler(CallbackQueryHandler(main_menu_handler, pattern='^menu_'))

    # This handler processes button clicks for monitoring
    application.add_handler(CallbackQueryHandler(button_callback, pattern='^(monitor|unmonitor)_'))

    # This handler processes button clicks for mode setting
    application.add_handler(CallbackQueryHandler(mode_callback, pattern='^mode_'))

    # This handler processes PnL button clicks
    application.add_handler(CallbackQueryHandler(pnl_callback, pattern='^pnl_'))

    # These handlers correspond to the buttons in the autotrade_menu
    application.add_handler(CallbackQueryHandler(autotrade_on, pattern='^autotrade_on$'))
    application.add_handler(CallbackQueryHandler(autotrade_off, pattern='^autotrade_off$'))

    # Set up the conversation handler for placing trades
    trade_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(trade_callback, pattern='^trade_')],
        states={
            ASKING_LOT_SIZE: [MessageHandler(filters.TEXT & ~filters.COMMAND, lot_size_input)],
        },
        fallbacks=[CommandHandler('cancel', cancel_conversation)],
        per_message=False # Make sure conversation is on a per-user basis
    )
    application.add_handler(trade_conv_handler)

    # Schedule the monitoring job
    job_queue = application.job_queue
    # Run the job every 5 minutes (300 seconds)
    job_queue.run_repeating(check_all_monitored_pairs_job, interval=300, first=10)
    logger.info("Scheduled monitoring job to run every 5 minutes.")

    # Start the Bot
    application.run_polling()
    logger.info("Bot has started successfully.")

    # Run the bot until the user presses Ctrl-C
    # application.run_until_disconnected() # This is for webhook based bots mostly

    # Shutdown MT5 connection when the bot is stopped
    # This part is tricky, as run_polling is blocking.
    # A proper shutdown sequence would require more complex signal handling.
    # For now, we rely on the user stopping the script.
    # mt5.shutdown()
    # logger.info("MetaTrader 5 connection shut down.")

if __name__ == '__main__':
    main()
