import numpy as np
import pandas as pd

DEFAULT_CONFIG = {
    "top": {
        "weights": (0.50, 0.20, 0.20, 0.10),
        "threshold": 0.60,
        "min_confidence": 0,
        "tp_atr": 1.0,
        "sl_atr": 1.5,
        "max_hold": 20,
    },
    "bottom": {
        "weights": (0.30, 0.25, 0.25, 0.20),
        "threshold": 0.55,
        "min_confidence": 10,
        "tp_atr": 1.0,
        "sl_atr": 1.5,
        "max_hold": 20,
    },
}

class ExhaustionDetector:
    def __init__(self, rsi_period=14, ema_period=20, atr_period=14, roc_period=3, config=None):
        self.rsi_period = rsi_period
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.roc_period = roc_period
        self.config = config or DEFAULT_CONFIG

    def _atr(self, df):
        h, l, c = df["high"], df["low"], df["close"]
        prev = c.shift(1)
        tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/self.atr_period, adjust=False).mean()

    def _rsi(self, series):
        d = series.diff()
        g = d.clip(lower=0).ewm(alpha=1/self.rsi_period, adjust=False).mean()
        l_ = (-d.clip(upper=0)).ewm(alpha=1/self.rsi_period, adjust=False).mean()
        rs = g / l_.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        rsi[(g > 0) & (l_ == 0)] = 100.0
        rsi[(g == 0) & (l_ > 0)] = 0.0
        rsi[(g == 0) & (l_ == 0)] = 50.0
        return rsi.fillna(50)

    def _compute_momentum_series(self, df, direction):
        cfg = self.config[direction]
        w_rsi, w_ema, w_body, w_roc = cfg["weights"]

        close = df["close"]
        rsi = self._rsi(close)
        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        ema_slope = ema.diff(3)
        ema_slope_prev = ema_slope.shift(3)
        body_ratio = ((df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)).fillna(0)
        body_3 = body_ratio.rolling(3, min_periods=1).mean()
        body_prev = body_ratio.shift(3).rolling(5, min_periods=1).mean()
        roc = close.pct_change(self.roc_period)
        roc_changes = roc.diff()
        hist_std = roc_changes.rolling(100, min_periods=10).std()

        rsi_5ago = rsi.shift(5)
        rsi_slope = rsi - rsi_5ago

        body_s = np.where(body_3 < body_prev * 0.8, 1.0, np.where(body_3 < body_prev, 0.5, 0.0))

        if direction == "top":
            rsi_strong = ((rsi > 60) & (rsi_slope < 0)).astype(float)
            rsi_weak = ((rsi_slope < 0) & (rsi <= 60)).astype(float) * 0.6
            rsi_s = rsi_strong + rsi_weak
            ema_s = ((ema_slope_prev > 0) & (ema_slope < ema_slope_prev)).astype(float)
            roc_norm = np.where(hist_std > 1e-10, -roc_changes / hist_std / 2.0, 0.0)
        else:
            rsi_strong = ((rsi < 40) & (rsi_slope > 0)).astype(float)
            rsi_weak = ((rsi_slope > 0) & (rsi >= 40)).astype(float) * 0.6
            rsi_s = rsi_strong + rsi_weak
            ema_s = ((ema_slope_prev < 0) & (ema_slope > ema_slope_prev)).astype(float)
            roc_norm = np.where(hist_std > 1e-10, roc_changes / hist_std / 2.0, 0.0)

        roc_s = np.clip(roc_norm, 0, 1)
        momentum = w_rsi * rsi_s + w_ema * ema_s + w_body * body_s + w_roc * roc_s

        return pd.Series(momentum, index=df.index), {
            "rsi": rsi, "rsi_s": pd.Series(rsi_s, index=df.index),
            "ema_s": pd.Series(ema_s, index=df.index),
            "body_s": pd.Series(body_s, index=df.index),
            "roc_s": pd.Series(roc_s, index=df.index),
        }

    def detect(self, df, direction):
        if len(df) < 30:
            return {"signal": False, "reason": "بيانات غير كافية"}
        momentum, comps = self._compute_momentum_series(df, direction)
        atr = self._atr(df)
        last_mom = float(momentum.iloc[-1])
        last_atr = float(atr.iloc[-1])
        last_close = float(df["close"].iloc[-1])
        cfg = self.config[direction]
        threshold = cfg["threshold"]

        if last_mom < threshold or last_atr <= 0:
            return {"signal": False, "momentum": last_mom, "reason": f"momentum {last_mom:.2f} < {threshold}"}

        conf_pre = min(100, int((last_mom - threshold) / (1 - threshold) * 100))
        min_conf = cfg.get("min_confidence", 0)
        if conf_pre < min_conf:
            return {"signal": False, "momentum": last_mom, "reason": f"confidence {conf_pre}% < min {min_conf}%"}

        if direction == "top":
            tp_price = last_close - cfg["tp_atr"] * last_atr
            sl_price = last_close + cfg["sl_atr"] * last_atr
            trade_direction = "SHORT"
        else:
            tp_price = last_close + cfg["tp_atr"] * last_atr
            sl_price = last_close - cfg["sl_atr"] * last_atr
            trade_direction = "LONG"

        return {
            "signal": True, "direction": trade_direction, "confidence": conf_pre,
            "momentum": round(last_mom, 3), "threshold": threshold,
            "entry_price": last_close, "tp_price": round(tp_price, 6),
            "sl_price": round(sl_price, 6), "atr": round(last_atr, 6),
            "max_hold_candles": cfg["max_hold"],
            "components": {
                "rsi": round(float(comps["rsi"].iloc[-1]), 2),
                "rsi_s": round(float(comps["rsi_s"].iloc[-1]), 3),
                "ema_s": round(float(comps["ema_s"].iloc[-1]), 3),
                "body_s": round(float(comps["body_s"].iloc[-1]), 3),
                "roc_s": round(float(comps["roc_s"].iloc[-1]), 3),
            },
        }

    def detect_both(self, df):
        top = self.detect(df, "top")
        bottom = self.detect(df, "bottom")
        signals = [s for s in (top, bottom) if s.get("signal")]
        if not signals:
            return {"signal": False, "top": top, "bottom": bottom}
        best = max(signals, key=lambda x: x["momentum"])
        return {"signal": True, "best": best, "top": top, "bottom": bottom}
