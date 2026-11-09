import sys, time, json, os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import ccxt
import httpx

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from exhaustion_detector import ExhaustionDetector
from live_trading import (
    open_position, close_position, sync_positions, load_state,
    is_halted, record_outcome, reset_halt, get_ex,
    count_open_trades, is_coin_already_open, get_all_open_symbols,
)

TG_TOKEN = os.getenv("PAPER_LAB_TG_TOKEN") or os.getenv("TELEGRAM_REPORT_BOT_TOKEN")
TG_CHAT = os.getenv("PAPER_LAB_TG_CHAT") or os.getenv("TELEGRAM_REPORT_CHAT_ID")

def send_tg(text, silent=False, keyboard=None):
    if not TG_TOKEN or not TG_CHAT: return False
    try:
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML", "disable_notification": silent}
        if keyboard: payload["reply_markup"] = {"inline_keyboard": keyboard}
        r = httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", json=payload, timeout=20)
        return r.status_code == 200
    except: return False

def esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)

def compute_supertrend(df, period=10, mult=3.0):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    if n < period + 2: return np.zeros(n, dtype=int)
    tr = np.zeros(n)
    for i in range(n):
        if i == 0: tr[i] = h[i] - l[i]
        else:
            pc = c[i-1]
            tr[i] = max(h[i]-l[i], abs(h[i]-pc), abs(l[i]-pc))
    atr = np.full(n, np.nan)
    if period >= n: return np.zeros(n, dtype=int)
    atr[period] = tr[1:period+1].sum() / period
    for i in range(period+1, n):
        atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
    hl2 = (h + l) / 2
    ub = np.full(n, np.nan); lb = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr[i]):
            ub[i] = hl2[i] + mult * atr[i]
            lb[i] = hl2[i] - mult * atr[i]
    direction = np.zeros(n, dtype=int)
    direction[period] = 1
    for i in range(period+1, n):
        if np.isnan(ub[i]) or np.isnan(lb[i]):
            direction[i] = direction[i-1]; continue
        if direction[i-1] == 1:
            if not np.isnan(lb[i-1]): lb[i] = max(lb[i], lb[i-1])
        else:
            if not np.isnan(ub[i-1]): ub[i] = min(ub[i], ub[i-1])
        prev_ub = ub[i-1] if not np.isnan(ub[i-1]) else ub[i]
        prev_lb = lb[i-1] if not np.isnan(lb[i-1]) else lb[i]
        if c[i] > prev_ub: direction[i] = 1
        elif c[i] < prev_lb: direction[i] = -1
        else: direction[i] = direction[i-1]
    return direction

def compute_adx(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    plus_dm = h.diff().where(h.diff() > -l.diff(), 0)
    minus_dm = -l.diff().where(-l.diff() > h.diff(), 0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()

COINS = ["BTC", "ETH", "BNB", "XRP", "SOL", "LTC", "LINK", "DOGE", "ADA", "AVAX", "DOT"]
MAX_CONCURRENT_TRADES = 5
SCAN_INTERVAL_SEC = 60
COOLDOWN_CANDLES = 5
CANDLES_LIMIT = 300
LIVE_TRADING = True
last_hourly_report_time = 0

LOG_DIR = BASE / "exhaustion_logs"
LOG_DIR.mkdir(exist_ok=True)
SIGNALS_FILE = LOG_DIR / "exhaustion_signals.csv"
PAPER_TRADES_FILE = LOG_DIR / "paper_trades.json"
PAUSE_FILE = LOG_DIR / "bot_paused.json"

LIVE_HOURS_WHITELIST = [
    (10, 3), (18, 3), (9, 0), (23, 2), (11, 1), (17, 0), (20, 3), (11, 0), (22, 3),
    (23, 1), (0, 2), (3, 2), (5, 1), (12, 1), (1, 6), (8, 1), (10, 1), (20, 1),
    (7, 0), (14, 0), (12, 0), (15, 6), (16, 3), (17, 6), (19, 6), (13, 2), (12, 3),
    (21, 0), (6, 5), (8, 0), (1, 1), (9, 5), (7, 6), (10, 2), (16, 1), (0, 5),
    (15, 0), (16, 5), (8, 2), (17, 1), (18, 6), (15, 3), (10, 6), (11, 6), (22, 6),
    (8, 6), (18, 2), (21, 3), (21, 2), (20, 6), (13, 0), (16, 2), (2, 2), (2, 0),
    (4, 1), (10, 0), (18, 5), (18, 0), (3, 6), (13, 3), (14, 2), (16, 4), (13, 6),
    (19, 0), (7, 1), (1, 2), (22, 4)
]

COIN_CONFIG = {
    "BTC":  {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 30},
    "ETH":  {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 40},
    "BNB":  {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 30},
    "XRP":  {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 30},
    "SOL":  {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 30},
    "LTC":  {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 40},
    "LINK": {"tp_atr": 2.0, "sl_atr": 2.5, "max_hold": 40},
    "DOGE": {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 30},
    "ADA":  {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 40},
    "AVAX": {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 40},
    "DOT":  {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 40},
}

def fetch_ohlcv(coin, tf, limit=CANDLES_LIMIT):
    ex = get_ex()
    sym = f"{coin}/USDT:USDT"
    if sym not in ex.markets: return None
    try:
        data = ex.fetch_ohlcv(sym, tf, limit=limit)
        if not data or len(data) < 50: return None
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("datetime").drop(columns=["ts"])
    except: return None

def fetch_5m(coin, limit=300):
    ex = get_ex()
    sym = f"{coin}/USDT:USDT"
    if sym not in ex.markets:
        sym = f"{coin}/USDT"
        if sym not in ex.markets: return None
    try:
        data = ex.fetch_ohlcv(sym, "5m", limit=limit)
        if not data or len(data) < 50: return None
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("datetime").drop(columns=["ts"])
    except: return None

def record_signal(coin, tf, sig, detector):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coin": coin, "tf": tf, "direction": sig["direction"],
        "confidence": sig["confidence"], "momentum": sig["momentum"],
        "entry": sig["entry_price"], "tp": sig["tp_price"], "sl": sig["sl_price"],
        "atr": sig["atr"], "max_hold_candles": sig["max_hold_candles"],
        "rsi": sig["components"]["rsi"], "rsi_s": sig["components"]["rsi_s"],
        "ema_s": sig["components"]["ema_s"], "body_s": sig["components"]["body_s"],
        "roc_s": sig["components"]["roc_s"],
    }
    df = pd.DataFrame([row])
    if SIGNALS_FILE.exists():
        existing = pd.read_csv(SIGNALS_FILE)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(SIGNALS_FILE, index=False)

def format_signal(coin, tf, sig):
    emoji = "🔻" if sig["direction"] == "SHORT" else "🔺"
    direction_ar = "بيع" if sig["direction"] == "SHORT" else "شراء"
    entry = sig['entry_price']; tp = sig['tp_price']; sl = sig['sl_price']
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour
    current_day = (now_utc.weekday() + 1) % 7
    is_white = (current_hour, current_day) in LIVE_HOURS_WHITELIST
    exec_type = "⚪ <b>[تداول حقيقي]</b>" if is_white else "📝 <b>[تداول ورقي فقط]</b>"
    return (
        f"{exec_type}\n{emoji} <b>{coin} {tf} — إشارة {direction_ar}</b>\n"
        f"━━━━━━━━━━━━━━━\n💪 الثقة: <b>{sig['confidence']}%</b> (momentum={sig['momentum']:.2f})\n"
        f"💵 الدخول: <code>{entry:.4f}</code>\n🎯 الهدف: <code>{tp:.4f}</code> ({(abs(tp-entry)/entry*100):.2f}%)\n"
        f"🛑 الوقف: <code>{sl:.4f}</code> ({(abs(sl-entry)/entry*100):.2f}%)\n"
        f"⏱️ أقصى انتظار: {sig['max_hold_candles']} شموع\n"
        f"📊 RSI={sig['components']['rsi']:.1f} | EMA_s={sig['components']['ema_s']:.2f} | Body_s={sig['components']['body_s']:.2f} | ROC_s={sig['components']['roc_s']:.2f}\n"
        f"🕐 {datetime.now().strftime('%H:%M')}"
    )

def record_paper_trade(coin, direction, entry, tp, sl, max_hold):
    trades = {}
    if PAPER_TRADES_FILE.exists():
        try: trades = json.loads(PAPER_TRADES_FILE.read_text())
        except: pass
    trade_id = f"{coin}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    trades[trade_id] = {
        "coin": coin, "direction": direction, "entry": entry, "tp": tp, "sl": sl,
        "max_hold": max_hold, "bars": 0, "opened_at": datetime.now(timezone.utc).isoformat(), "status": "open"
    }
    PAPER_TRADES_FILE.write_text(json.dumps(trades, indent=2))
    log(f"  📝 [PAPER] تم تسجيل صفقة ورقية: {coin} {direction}")

def check_paper_trades(df_1h):
    if not PAPER_TRADES_FILE.exists(): return
    try: trades = json.loads(PAPER_TRADES_FILE.read_text())
    except: return
    changed = False
    ex = get_ex()
    for tid, t in list(trades.items()):
        if t["status"] != "open": continue
        try:
            sym = f"{t['coin']}/USDT:USDT"
            ticker = ex.fetch_ticker(sym)
            current_price = float(ticker["last"])
        except: continue
        t["bars"] += 1
        outcome = None; close_price = None
        if t["direction"] == "LONG":
            if current_price <= t["sl"]: outcome, close_price = "LOSS", t["sl"]
            elif current_price >= t["tp"]: outcome, close_price = "WIN", t["tp"]
        else:
            if current_price >= t["sl"]: outcome, close_price = "LOSS", t["sl"]
            elif current_price <= t["tp"]: outcome, close_price = "WIN", t["tp"]
        if not outcome and t["bars"] >= t["max_hold"]:
            outcome = "EXPIRED"; close_price = current_price
        if outcome:
            t["status"] = "closed"; t["outcome"] = outcome; t["close_price"] = close_price
            t["closed_at"] = datetime.now(timezone.utc).isoformat()
            changed = True
            emoji = "✅" if outcome == "WIN" else ("❌" if outcome == "LOSS" else "⏰")
            send_tg(f"{emoji} <b>إغلاق صفقة ورقية</b>\n🪙 {t['coin']}\n💵 {t['entry']:.4f} → {close_price:.4f}\n🏁 {outcome}")
    if changed: PAPER_TRADES_FILE.write_text(json.dumps(trades, indent=2))

def is_bot_paused():
    if PAUSE_FILE.exists():
        try: return json.loads(PAUSE_FILE.read_text()).get("paused", False)
        except: return False
    return False

def set_bot_paused(paused):
    PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAUSE_FILE.write_text(json.dumps({"paused": paused}), encoding="utf-8")

def send_main_menu():
    halted, consec = is_halted()
    open_count = 0
    try:
        ex = get_ex()
        open_count = sum(1 for p in ex.fetch_positions() if float(p.get("contracts", 0) or 0) > 0)
    except: pass
    text = f"🤖 <b>ExhaustionBot — لوحة التحكم</b>\n━━━━━━━━━━━━━━━━\n"
    text += f"الحالة: {'🟢 يعمل' if not is_bot_paused() else '⏸️ موقوف'}\n"
    if halted: text += f"
