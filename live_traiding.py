import sys, time, json
from pathlib import Path
from datetime import datetime, timezone
import ccxt

# ═══ المسار ليعمل على Render ═══
BASE = Path(__file__).parent
STATE_FILE = BASE / "exhaustion_logs" / "live_positions.json"
LOSSES_FILE = BASE / "exhaustion_logs" / "consecutive_losses.json"
LOG_FILE = BASE / "exhaustion_logs" / "live_trading.log"
MAX_CONSECUTIVE_LOSSES = 2

# ═══ قراءة المفاتيح من متغيرات البيئة (Render) ═══
env = {}
if (BASE / ".env").exists():
    for line in (BASE / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

import os
API_KEY = os.getenv("BINANCE_API_KEY") or env.get("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET") or env.get("BINANCE_API_SECRET")
POSITION_SIZE_USDT = 100.0
LEVERAGE = 50

# ═══ دوال التسجيل ═══
def log(msg):
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except: pass

# ═══ الاتصال بـ Binance ═══
_ex = None
def get_ex():
    global _ex
    if _ex is None:
        _ex = ccxt.binance({
            "apiKey": API_KEY, "secret": API_SECRET,
            "enableRateLimit": True, "timeout": 30000,
            "options": {"defaultType": "future", "adjustForTimeDifference": True},
        })
        _ex.load_markets()
    return _ex

def load_state():
    if STATE_FILE.exists():
        try: return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except: return {"positions": {}}
    return {"positions": {}}

def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

def load_losses():
    if LOSSES_FILE.exists():
        try: return json.loads(LOSSES_FILE.read_text(encoding="utf-8"))
        except: pass
    return {"consecutive": 0, "halted": False, "last_update": None, "history": []}

def save_losses(data):
    LOSSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["last_update"] = datetime.now(timezone.utc).isoformat()
    LOSSES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def record_outcome(coin, outcome, pnl_pct=0.0):
    data = load_losses()
    if outcome == "loss": data["consecutive"] = int(data.get("consecutive", 0)) + 1
    elif outcome == "win": data["consecutive"] = 0
    data.setdefault("history", []).append({"coin": coin, "outcome": outcome, "pnl_pct": pnl_pct, "ts": datetime.now(timezone.utc).isoformat()})
    data["history"] = data["history"][-50:]
    if data["consecutive"] >= MAX_CONSECUTIVE_LOSSES:
        data["halted"] = True
    save_losses(data)

def is_halted():
    data = load_losses()
    return bool(data.get("halted", False)), int(data.get("consecutive", 0))

def reset_halt():
    data = load_losses()
    data["halted"] = False
    data["consecutive"] = 0
    save_losses(data)

def symbol(coin): return f"{coin}/USDT:USDT"

def amount(price, notional=POSITION_SIZE_USDT):
    return round(notional / price, 6)

def open_position(coin, direction, entry_price, tp_price, sl_price, confidence, momentum):
    ex = get_ex()
    sym = symbol(coin)
    if sym not in ex.markets: return None
    
    halted, _ = is_halted()
    if halted: return None
    
    state = load_state()
    if coin in state["positions"]: return None
    
    try:
        try: ex.set_leverage(LEVERAGE, sym)
        except: pass
        
        qty = amount(entry_price)
        side = "buy" if direction == "LONG" else "sell"
        pos_side = "LONG" if direction == "LONG" else "SHORT"
        
        order = ex.create_order(sym, "market", side, qty, params={"positionSide": "BOTH"})
        
        time.sleep(1)
        positions = ex.fetch_positions([sym])
        is_open = any(p["symbol"] == sym and float(p.get("contracts", 0) or 0) > 0 for p in positions)
        if not is_open: return None
        
        tp_side = "sell" if direction == "LONG" else "buy"
        try: ex.create_order(sym, "TAKE_PROFIT_MARKET", tp_side, qty, None, {"stopPrice": tp_price, "reduceOnly": True, "positionSide": "BOTH"})
        except: pass
        try: ex.create_order(sym, "STOP_MARKET", tp_side, qty, None, {"stopPrice": sl_price, "reduceOnly": True, "positionSide": "BOTH"})
        except: pass
        
        state["positions"][coin] = {
            "direction": direction, "entry": entry_price, "tp": tp_price, "sl": sl_price,
            "qty": qty, "confidence": confidence, "momentum": momentum,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state)
        return state["positions"][coin]
    except Exception as e:
        log(f"❌ open_position {coin}: {e}")
        return None

def close_position(coin, reason="manual"):
    ex = get_ex()
    sym = symbol(coin)
    state = load_state()
    if coin not in state["positions"]: return False
    try:
        positions = ex.fetch_positions([sym])
        for p in positions:
            if p["symbol"] == sym and float(p.get("contracts", 0) or 0) > 0:
                qty = float(p["contracts"])
                side = "sell" if p["side"] == "long" else "buy"
                ex.create_order(sym, "market", side, qty, params={"positionSide": "BOTH", "reduceOnly": True})
                break
        try:
            for o in ex.fetch_open_orders(sym): ex.cancel_order(o["id"], sym)
        except: pass
        del state["positions"][coin]
        save_state(state)
        return True
    except: return False

def sync_positions():
    ex = get_ex()
    state = load_state()
    try:
        positions = ex.fetch_positions()
        active = {p["symbol"].split("/")[0]: p for p in positions if p.get("symbol","").endswith(":USDT") and float(p.get("contracts", 0) or 0) > 0}
        closed = []
        for coin in list(state["positions"].keys()):
            if coin not in active:
                pos_info = state["positions"][coin]
                try:
                    ticker = ex.fetch_ticker(symbol(coin))
                    last_price = float(ticker["last"])
                except: last_price = pos_info["entry"]
                entry = float(pos_info["entry"])
                direction = pos_info["direction"]
                tp = float(pos_info["tp"]); sl = float(pos_info["sl"])
                dist_tp = abs(last_price - tp); dist_sl = abs(last_price - sl)
                outcome = "win" if dist_tp < dist_sl else "loss"
                pnl_pct = ((last_price - entry) / entry * 100) if direction == "LONG" else ((entry - last_price) / entry * 100)
                record_outcome(coin, outcome, pnl_pct)
                closed.append(coin)
                del state["positions"][coin]
        if closed: save_state(state)
        return list(active.keys())
    except: return list(state["positions"].keys())

def count_open_trades():
    try:
        ex = get_ex()
        return sum(1 for p in ex.fetch_positions() if float(p.get("contracts", 0) or 0) > 0)
    except: return 0

def get_all_open_symbols():
    try:
        ex = get_ex()
        return [p["symbol"].split("/")[0] for p in ex.fetch_positions() if float(p.get("contracts", 0) or 0) > 0]
    except: return []

def is_coin_already_open(coin):
    return coin in get_all_open_symbols()
ENDOFFILE
