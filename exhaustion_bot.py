
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import pandas as pd
import ccxt
import httpx

from exhaustion_detector import ExhaustionDetector

# ═══ استيراد موحد من live_trading ═══
from live_trading import (
    open_position, close_position, sync_positions, load_state,
    is_halted, record_outcome, reset_halt, get_ex,
    count_open_trades, is_coin_already_open, get_all_open_symbols,
)
def compute_adx(df, period=14):
    import pandas as pd
    h, l, c = df['high'], df['low'], df['close']
    plus_dm = h.diff().where(h.diff() > -l.diff(), 0)
    minus_dm = -l.diff().where(-l.diff() > h.diff(), 0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()
  

# ═══ Supertrend (فلتر اتجاه) ═══
def compute_supertrend(df, period=10, mult=3.0):
    import numpy as np
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

# ═══ الإعدادات ═══
COINS = ["BTC", "ETH", "BNB", "XRP", "SOL", "LTC", "LINK", "DOGE", "ADA", "AVAX", "DOT"]
MAX_CONCURRENT_TRADES = 5
TIMEFRAMES = ["1h"]
SCAN_INTERVAL_SEC = 60
COOLDOWN_CANDLES = 5
CANDLES_LIMIT = 300
LIVE_TRADING = True

# قائمة النوافذ البيضاء (67 نافذة)
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

BASE = Path("/root/bot2")
LOG_DIR = BASE / "exhaustion_logs"
LOG_DIR.mkdir(exist_ok=True)
SIGNALS_FILE = LOG_DIR / "exhaustion_signals.csv"
PAPER_TRADES_FILE = LOG_DIR / "paper_trades.json"
PAUSE_FILE = LOG_DIR / "bot_paused.json"
last_hourly_report_time = 0

env = {}
if (BASE / ".env").exists():
    for line in (BASE / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
TG_TOKEN = env.get("PAPER_LAB_TG_TOKEN") or env.get("TELEGRAM_REPORT_BOT_TOKEN")
TG_CHAT = env.get("PAPER_LAB_TG_CHAT") or env.get("TELEGRAM_REPORT_CHAT_ID")

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
        "rsi": sig["components"]["rsi"],
        "rsi_s": sig["components"]["rsi_s"],
        "ema_s": sig["components"]["ema_s"],
        "body_s": sig["components"]["body_s"],
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
    entry = sig['entry_price']
    tp = sig['tp_price']
    sl = sig['sl_price']
    
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour
    current_day = (now_utc.weekday() + 1) % 7
    is_white = (current_hour, current_day) in LIVE_HOURS_WHITELIST
    exec_type = "⚪ <b>[تداول حقيقي]</b>" if is_white else "📝 <b>[تداول ورقي فقط]</b>"
    
    return (
        f"{exec_type}\n"
        f"{emoji} <b>{coin} {tf} — إشارة {direction_ar}</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"💪 الثقة: <b>{sig['confidence']}%</b> (momentum={sig['momentum']:.2f})\n"
        f"💵 الدخول: <code>{entry:.4f}</code>\n"
        f"🎯 الهدف: <code>{tp:.4f}</code> ({(abs(tp-entry)/entry*100):.2f}%)\n"
        f"🛑 الوقف: <code>{sl:.4f}</code> ({(abs(sl-entry)/entry*100):.2f}%)\n"
        f"⏱️ أقصى انتظار: {sig['max_hold_candles']} شموع\n"
        f"📊 RSI={sig['components']['rsi']:.1f} | EMA_s={sig['components']['ema_s']:.2f} | Body_s={sig['components']['body_s']:.2f} | ROC_s={sig['components']['roc_s']:.2f}\n"
        f"🕐 {datetime.now().strftime('%H:%M')}"
    )

def record_paper_trade(coin, direction, entry, tp, sl, max_hold):
    import json
    trades = {}
    if PAPER_TRADES_FILE.exists():
        try: trades = json.loads(PAPER_TRADES_FILE.read_text())
        except: pass
    trade_id = f"{coin}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    trades[trade_id] = {
        "coin": coin, "direction": direction, "entry": entry,
        "tp": tp, "sl": sl, "max_hold": max_hold, "bars": 0,
        "opened_at": datetime.now(timezone.utc).isoformat(), "status": "open"
    }
    PAPER_TRADES_FILE.write_text(json.dumps(trades, indent=2))
    log(f"  📝 [PAPER] تم تسجيل صفقة ورقية: {coin} {direction}")

def check_paper_trades(df_1h):
    import json
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
            msg = (
                f"{emoji} <b>إغلاق صفقة ورقية</b>\n"
                f"━━━━━━━━━━━━━━━\n"
                f"🪙 العملة: <b>{t['coin']}</b>\n"
                f"📈 الاتجاه: {t['direction']}\n"
                f"💵 الدخول: {t['entry']:.4f} | الخروج: {close_price:.4f}\n"
                f"🏁 النتيجة: <b>{outcome}</b>\n"
                f"⏱️ المدة: {t['bars']} شمعة\n"
            )
            send_tg(msg)
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
        positions = ex.fetch_positions()
        open_count = sum(1 for p in positions if float(p.get("contracts", 0) or 0) > 0)
    except: pass
    text = f"🤖 <b>ExhaustionBot — لوحة التحكم</b>\n━━━━━━━━━━━━━━━━\n"
    text += f"الحالة: {'🟢 يعمل' if not is_bot_paused() else '⏸️ موقوف'}\n"
    if halted: text += f"🛑 <b>التداول موقوف</b> — {consec} خسائر متتالية\n"
    text += f"💰 الصفقات المفتوحة: {open_count}\nاختر إجراءً:"
    paused = is_bot_paused()
    btn_pause = {"text": "▶️ تشغيل البوت" if paused else "⏸️ إيقاف البوت", "callback_data": "menu_run" if paused else "menu_stop"}
    kb = [
        [btn_pause, {"text": "📊 الحالة", "callback_data": "menu_status"}],
        [{"text": "🔓 استئناف بعد الخسائر", "callback_data": "menu_reset"}, {"text": "💰 الصفقات المفتوحة", "callback_data": "menu_positions"}],
        [{"text": "📈 الرصيد", "callback_data": "balance"}, {"text": "🚨 إغلاق كل الصفقات", "callback_data": "menu_close_all"}],
        [{"text": "📋 تقرير اليوم", "callback_data": "menu_daily"}, {"text": "🔄 تحديث", "callback_data": "menu_refresh"}]
    ]
    send_tg(text, keyboard=kb)

def handle_message(msg_text):
    t = msg_text.strip().lower()
    if t in ("/start", "/menu", "menu", "القائمة"):
        send_main_menu()

def tg_answer_callback(cb_id, text=""):
    try: httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/answerCallbackQuery", json={"callback_query_id": cb_id, "text": text}, timeout=5)
    except: pass

def tg_edit_message(msg_id, text, keyboard=None):
    try:
        payload = {"chat_id": TG_CHAT, "message_id": msg_id, "text": text, "parse_mode": "HTML"}
        if keyboard: payload["reply_markup"] = {"inline_keyboard": keyboard}
        httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/editMessageText", json=payload, timeout=10)
    except: pass

def handle_callback(cb_data, cb_id, msg_id):
    if cb_data == "balance":
        try:
            from live_trading import get_ex
            ex = get_ex()
            usdt = ex.fetch_balance().get("USDT", {})
            txt = f"📈 <b>الرصيد:</b>\nالمتاح: {usdt.get('free', 0):.2f} USDT\nالإجمالي: {usdt.get('total', 0):.2f} USDT"
        except Exception as e: txt = f"❌ خطأ: {e}"
        tg_answer_callback(cb_id, "الرصيد")
        tg_edit_message(msg_id, txt)
    elif cb_data == "menu_status":
        halted, consec = is_halted()
        try:
            ex = get_ex(); positions = ex.fetch_positions()
            count = sum(1 for p in positions if float(p.get("contracts", 0) or 0) > 0)
            txt = f"📊 <b>الحالة:</b> {'🛑 موقوف' if halted else '🟢 نشط'}\nالخسائر المتتالية: {consec}\nالصفقات المفتوحة: {count}"
        except: txt = f"📊 <b>الحالة:</b> {'🛑 موقوف' if halted else '🟢 نشط'}"
        tg_edit_message(msg_id, txt)
    elif cb_data == "menu_positions":
        try:
            ex = get_ex(); positions = ex.fetch_positions()
            active = [p for p in positions if float(p.get("contracts", 0) or 0) > 0]
            if not active: txt = "لا توجد صفقات مفتوحة."
            else:
                txt = "💼 <b>الصفقات المفتوحة:</b>\n"
                for p in active:
                    coin = p["symbol"].split("/")[0]
                    pnl = float(p.get("unrealizedPnl", 0) or 0)
                    txt += f"- {coin}: {p['side']} {p['contracts']} @ {p['entryPrice']} | PnL: {pnl:+.2f}\n"
        except Exception as e: txt = f"❌ خطأ: {e}"
        tg_edit_message(msg_id, txt)
    elif cb_data == "menu_reset":
        reset_halt()
        tg_edit_message(msg_id, "🔓 تم استئناف التداول.")
        send_main_menu()
    elif cb_data == "menu_run":
        set_bot_paused(False); tg_edit_message(msg_id, "▶️ تم التشغيل."); send_main_menu()
    elif cb_data == "menu_stop":
        set_bot_paused(True); tg_edit_message(msg_id, "⏸️ تم الإيقاف."); send_main_menu()
    elif cb_data == "menu_close_all":
        state = load_state()
        pos = list(state.get('positions', {}).keys())
        for coin in pos: close_position(coin, reason="telegram_close_all")
        tg_edit_message(msg_id, f"🚨 تم إغلاق {len(pos)} صفقة.")
        send_main_menu()
    elif cb_data == "menu_daily":
        losses = load_losses() if 'load_losses' in globals() else {}
        hist = losses.get("history", [])
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_hist = [h for h in hist if h.get("ts", "").startswith(today)]
        w = sum(1 for h in today_hist if h["outcome"] == "win")
        l = sum(1 for h in today_hist if h["outcome"] == "loss")
        pnl = sum(h.get("pnl_pct", 0) for h in today_hist)
        wr = w/(w+l)*100 if (w+l) > 0 else 0
        txt = f"📋 <b>تقرير اليوم ({today})</b>\nالصفقات: {len(today_hist)}\n✅ ربح: {w} | ❌ خسارة: {l}\nWR: {wr:.1f}%\nPnL: {pnl:+.2f}%"
        tg_edit_message(msg_id, txt)
    elif cb_data == "menu_refresh":
        send_main_menu()

def scan_once(detector, state):
    found = 0
    if is_bot_paused(): return 0
    try:
        df_1h_btc = fetch_ohlcv("BTC", "1h")
        if df_1h_btc is not None: check_paper_trades(df_1h_btc.iloc[:-1])
    except: pass
    
    if LIVE_TRADING:
        halted, consec = is_halted()
        if halted:
            log(f"🛑 التداول موقوف ({consec} خسائر) — لا فحص جديد")
            try: sync_positions()
            except: pass
            return 0

    for coin in COINS:
        for tf in TIMEFRAMES:
            df = fetch_ohlcv(coin, tf)
            if df is None or len(df) < 50: continue
            df_closed = df.iloc[:-1]
            if len(df_closed) < 30: continue
            for direction in ("top", "bottom"):
                sig = detector.detect(df_closed, direction)
                if not sig.get("signal"): continue
                try:
                    df_5m = fetch_5m(coin)
                    if df_5m is None: continue
                    df_5m_closed = df_5m.iloc[:-1]
                    if len(df_5m_closed) < 20: continue
                    st_trend = compute_supertrend(df_5m_closed)
                    st_val = int(st_trend[-1]) if len(st_trend) > 0 else 0
                    is_aligned = ((direction == "top" and st_val == -1) or (direction == "bottom" and st_val == 1))
                    if not is_aligned: continue
                except: continue

                key = f"{coin}_{tf}_{direction}"
                now_ts = df_closed.index[-1].isoformat()
                if state["last_signals"].get(key) == now_ts: continue
                last_ts = state["last_signals"].get(key)
                if last_ts:
                    try:
                        hours = (pd.to_datetime(now_ts) - pd.to_datetime(last_ts)).total_seconds() / 3600
                        if hours < COOLDOWN_CANDLES: continue
                    except: pass

                log(f"  🎯 {coin} {tf} {sig['direction']} | conf={sig['confidence']}% | mom={sig['momentum']}")
                kb = [
                    [{"text": "🛑 إغلاق الصفقة", "callback_data": f"close_{coin}"}, {"text": "📊 تحديث الحالة", "callback_data": f"status_{coin}"}],
                    [{"text": "📈 الرصيد", "callback_data": "balance"}, {"text": "🔄 تحديث الإشارة", "callback_data": f"refresh_{coin}"}]
                ]

                execution_success = False
                if LIVE_TRADING:
                    now_utc = datetime.now(timezone.utc)
                    current_hour = now_utc.hour
                    current_day = (now_utc.weekday() + 1) % 7
                    is_profitable_window = (current_hour, current_day) in LIVE_HOURS_WHITELIST
                    if is_profitable_window:
                        try:
                            adx_series = compute_adx(df_closed, 14)
                            adx_val = adx_series.iloc[-1] if not pd.isna(adx_series.iloc[-1]) else 0
                            if adx_val > 30:
                                log(f"  🚫 مرفوضة (ADX {adx_val:.1f} > 30) — تحويل إلى ورقي")
                                cfg = COIN_CONFIG.get(coin, {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 30})
                                entry = float(sig["entry_price"]); atr = float(sig["atr"])
                                tp_p = entry + cfg["tp_atr"]*atr if sig["direction"]=="LONG" else entry - cfg["tp_atr"]*atr
                                sl_p = entry - cfg["sl_atr"]*atr if sig["direction"]=="LONG" else entry + cfg["sl_atr"]*atr
                                record_paper_trade(coin, sig["direction"], entry, round(tp_p,6), round(sl_p,6), cfg["max_hold"])
                                send_tg(f"🚫 <b>رفض صفقة حقيقية (ADX مرتفع)</b>\n🪙 {coin}\n📊 ADX: {adx_val:.1f} (> 30)\n📝 تم التحويل إلى التنفيذ الورقي")
                                state["last_signals"][key] = now_ts
                                execution_success = True
                                break
                        except Exception as e: log(f"  ⚠️ ADX: {e}")
                        try:
                            if is_coin_already_open(coin):
                                log(f"  🚫 مرفوضة (مفتوحة بالفعل) — تحويل إلى ورقي")
                                cfg = COIN_CONFIG.get(coin, {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 30})
                                entry = float(sig["entry_price"]); atr = float(sig["atr"])
                                tp_p = entry + cfg["tp_atr"]*atr if sig["direction"]=="LONG" else entry - cfg["tp_atr"]*atr
                                sl_p = entry - cfg["sl_atr"]*atr if sig["direction"]=="LONG" else entry + cfg["sl_atr"]*atr
                                record_paper_trade(coin, sig["direction"], entry, round(tp_p,6), round(sl_p,6), cfg["max_hold"])
                                send_tg(f"🚫 <b>رفض صفقة حقيقية (مفتوحة بالفعل)</b>\n🪙 {coin}\n📝 تم التحويل إلى التنفيذ الورقي")
                                state["last_signals"][key] = now_ts
                                execution_success = True
                                break
                            if count_open_trades() >= MAX_CONCURRENT_TRADES:
                                log(f"  🚫 مرفوضة (تجاوز الحد) — تحويل إلى ورقي")
                                cfg = COIN_CONFIG.get(coin, {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 30})
                                entry = float(sig["entry_price"]); atr = float(sig["atr"])
                                tp_p = entry + cfg["tp_atr"]*atr if sig["direction"]=="LONG" else entry - cfg["tp_atr"]*atr
                                sl_p = entry - cfg["sl_atr"]*atr if sig["direction"]=="LONG" else entry + cfg["sl_atr"]*atr
                                record_paper_trade(coin, sig["direction"], entry, round(tp_p,6), round(sl_p,6), cfg["max_hold"])
                                send_tg(f"🚫 <b>رفض صفقة حقيقية (تجاوز الحد)</b>\n🪙 {coin}\n📝 تم التحويل إلى التنفيذ الورقي")
                                state["last_signals"][key] = now_ts
                                execution_success = True
                                break
                        except Exception as e: log(f"  ⚠️ فحص الصفقات: {e}")
                        
                        try:
                            cfg = COIN_CONFIG.get(coin, {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 30})
                            entry = float(sig["entry_price"]); atr = float(sig["atr"])
                            tp_price = entry + cfg["tp_atr"]*atr if sig["direction"]=="LONG" else entry - cfg["tp_atr"]*atr
                            sl_price = entry - cfg["sl_atr"]*atr if sig["direction"]=="LONG" else entry + cfg["sl_atr"]*atr
                            sig['tp_price'] = round(tp_price, 6); sig['sl_price'] = round(sl_price, 6)
                            result = open_position(coin, sig["direction"], entry, round(tp_price, 6), round(sl_price, 6), sig["confidence"], sig["momentum"])
                            if result:
                                log(f"  💰 [LIVE] تم فتح صفقة حقيقية: {coin}")
                                execution_success = True
                        except Exception as e: log(f"  ❌ live open: {e}")
                    else:
                        log(f"  📝 [PAPER] فتح صفقة ورقية (نافذة سوداء)")
                        cfg = COIN_CONFIG.get(coin, {"tp_atr": 2.0, "sl_atr": 2.0, "max_hold": 30})
                        entry = float(sig["entry_price"]); atr = float(sig["atr"])
                        tp_p = entry + cfg["tp_atr"]*atr if sig["direction"]=="LONG" else entry - cfg["tp_atr"]*atr
                        sl_p = entry - cfg["sl_atr"]*atr if sig["direction"]=="LONG" else entry + cfg["sl_atr"]*atr
                        record_paper_trade(coin, sig["direction"], entry, round(tp_p,6), round(sl_p,6), cfg["max_hold"])
                        execution_success = True
                
                if execution_success:
                    try: send_tg(format_signal(coin, tf, sig), keyboard=kb)
                    except Exception as e: log(f"  ⚠️ TG: {e}")
                    record_signal(coin, tf, sig, detector)
                    state["last_signals"][key] = now_ts
                    found += 1
    if LIVE_TRADING:
        try: sync_positions()
        except: pass
    return found

def heartbeat_loop():
    log("❤️ نبضة كل ساعة بدأت")
    while True:
        try:
            now = datetime.now(timezone.utc)
            current_hour = now.hour
            current_day = (now.weekday() + 1) % 7
            day_names = ['الأحد', 'الاثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت']
            day_name = day_names[current_day]
            is_white = (current_hour, current_day) in LIVE_HOURS_WHITELIST
            status = "⚪ <b>نافذة بيضاء</b> (تداول حقيقي)" if is_white else "⚫ <b>نافذة سوداء</b> (تداول ورقي)"
            next_hour = (current_hour + 1) % 24
            next_day = current_day if next_hour > 0 else (current_day + 1) % 7
            is_next_white = (next_hour, next_day) in LIVE_HOURS_WHITELIST
            next_status = "⚪ <b>بيضاء</b>" if is_next_white else "⚫ <b>سوداء</b>"
            msg = f"⏰ <b>نبضة الساعة</b>\n━━━━━━━━━━━━━━━\n🕐 الحالية: <b>{current_hour:02d}:00</b> ({day_name}) | {status}\n⏳ القادمة: <b>{next_hour:02d}:00</b> | {next_status}"
            send_tg(msg, silent=True)
            seconds_into_hour = now.minute * 60 + now.second
            sleep_seconds = 3600 - seconds_into_hour
            if sleep_seconds < 10: sleep_seconds = 3600
            time.sleep(sleep_seconds)
        except Exception as e:
            log(f"❌ خطأ في النبضة: {e}")
            time.sleep(60)

def send_summary(detector):
    if not SIGNALS_FILE.exists(): return
    df = pd.read_csv(SIGNALS_FILE)
    now = datetime.now(timezone.utc)
    day_ago = (now - pd.Timedelta(hours=24)).isoformat()
    recent = df[df["timestamp"] > day_ago] if "timestamp" in df else df
    today_n = len(recent)
    if today_n == 0: return
    long_n = len(recent[recent["direction"] == "LONG"]) if today_n else 0
    short_n = len(recent[recent["direction"] == "SHORT"]) if today_n else 0
    msg = f"📊 <b>تقرير الساعة</b>\n━━━━━━━━━━━━━━━\n📈 إشارات (24س): {today_n} (LONG: {long_n} | SHORT: {short_n})\n🕐 {now.strftime('%H:%M')}"
    send_tg(msg, silent=True)

def tg_polling_loop():
    offset = 0
    log("🤖 مستمع أزرار تلغرام بدأ")
    while True:
        try:
            r = httpx.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", params={"offset": offset, "timeout": 20}, timeout=25)
            if r.status_code == 200:
                for upd in r.json().get("result", []):
                    offset = upd["update_id"] + 1
                    if "callback_query" in upd:
                        cb = upd["callback_query"]
                        handle_callback(cb.get("data", ""), cb["id"], cb["message"]["message_id"])
                    elif "message" in upd:
                        msg = upd["message"]
                        if "text" in msg: handle_message(msg["text"])
        except: time.sleep(5)
        time.sleep(1)

def main():
    import threading
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    threading.Thread(target=tg_polling_loop, daemon=True).start()
    
    log("🚀 ExhaustionBot v5.0 بدأ")
    detector = ExhaustionDetector()
    state = {"last_signals": {}, "last_rejected": {}}
    scans = 0
    while True:
        scans += 1
        log(f"── دورة #{scans} ──")
        try:
            found = scan_once(detector, state)
            log(f"✅ دورة #{scans}: {found} إشارة جديدة")
            global last_hourly_report_time
            if time.time() - last_hourly_report_time >= 3600:
                send_summary(detector)
                last_hourly_report_time = time.time()
        except Exception as e:
            log(f"❌ خطأ: {e}")
            send_tg(f"⚠️ <b>ExhaustionBot خطأ</b>\n<code>{str(e)[:200]}</code>")
        time.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    main()
