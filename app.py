"""
╔══════════════════════════════════════════════════════════════╗
║        بوت المضاربة السريعة — MEXC Scalping Bot v3.1        ║
║         إصلاح مشكلة session_state في الخيوط المنفصلة        ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ════════════════════════════════════════════════════════════
#  متغيرات عالمية مشتركة بين الخيط والواجهة
#  (الحل الأساسي للمشكلة)
# ════════════════════════════════════════════════════════════
_lock = threading.Lock()

_state = {
    "running":         False,
    "balance":         20.0,
    "initial_capital": 20.0,
    "withdrawn":       0.0,
    "withdraw_ready":  False,
    "compound_log":    [],
    "trades":          [],
    "open_trades":     {},
    "logs":            deque(maxlen=120),
    "scan_results":    deque(maxlen=50),
    "alerts":          deque(maxlen=5),
    "ai_model":        None,
    "ai_scaler":       None,
    "ai_trained":      False,
    "ai_history":      [],
    "all_symbols":     [],
    "symbols_loaded":  False,
    "current_scan":    "",
    "scan_cycle":      0,
    "last_update":     None,
    "scan_list":       [],
    "stats": {
        "win":0,"loss":0,"pnl_today":0.0,"pnl_total":0.0,
        "total_fees":0.0,"best_trade":0.0,"worst_trade":0.0,
        "moon_shots":0,"total_scanned":0,
    },
}

_stop_event = threading.Event()

def get(key):
    with _lock:
        return _state[key]

def set_(key, value):
    with _lock:
        _state[key] = value

def get_stat(key):
    with _lock:
        return _state["stats"][key]

def set_stat(key, value):
    with _lock:
        _state["stats"][key] = value

# ════════════════════════════════════════════════════════════
#  إعداد الصفحة
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="بوت المضاربة v3.1",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Cairo:wght@400;600;700;900&display=swap');
:root {
    --bg:     #060b10;
    --card:   #0b1520;
    --panel:  #0f1e2e;
    --border: #1a3050;
    --accent: #00c8ff;
    --green:  #00ff9d;
    --red:    #ff3860;
    --yellow: #ffd600;
    --orange: #ff8c00;
    --purple: #c084fc;
    --text:   #c8ddf0;
    --dim:    #4a6a8a;
}
html,body,[data-testid="stAppViewContainer"]{
    background:var(--bg)!important;
    color:var(--text)!important;
    font-family:'Cairo',sans-serif!important;
}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stSidebar"]{
    background:var(--card)!important;
    border-left:1px solid var(--border)!important;
}
.block-container{padding:1rem 1.5rem!important;}
.mc{background:var(--card);border:1px solid var(--border);
    border-radius:10px;padding:14px 18px;position:relative;
    overflow:hidden;margin-bottom:8px;}
.mc::after{content:'';position:absolute;bottom:0;right:0;
    width:100%;height:3px;background:var(--accent);}
.mc.g::after{background:var(--green);}
.mc.r::after{background:var(--red);}
.mc.y::after{background:var(--yellow);}
.mc.o::after{background:var(--orange);}
.mc.p::after{background:var(--purple);}
.ml{font-size:10px;color:var(--dim);margin-bottom:5px;font-weight:600;}
.mv{font-size:22px;font-weight:900;font-family:'JetBrains Mono',monospace;}
.ms{font-size:11px;color:var(--dim);margin-top:4px;}
.sh{font-family:'Cairo',sans-serif;font-size:14px;font-weight:700;
    color:var(--accent);border-bottom:1px solid var(--border);
    padding-bottom:8px;margin:20px 0 12px;}
.sc{background:var(--panel);border:1px solid var(--border);
    border-radius:10px;padding:12px 16px;
    display:flex;align-items:center;justify-content:space-between;
    margin-bottom:5px;}
.sc.a{border-color:var(--accent);}
.sc.p{border-color:var(--green);}
.sc.f{opacity:.65;}
.sc.moon{border-color:var(--purple);background:#1a0a2e;}
.sym{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700;}
.rsn{font-size:11px;color:var(--dim);margin-top:2px;}
.badge{font-size:10px;padding:3px 9px;border-radius:20px;font-weight:700;}
.b-scan{background:#0a2040;color:var(--accent);border:1px solid var(--accent);}
.b-pass{background:#002a1a;color:var(--green);border:1px solid var(--green);}
.b-fail{background:#1a0010;color:var(--red);border:1px solid var(--red);}
.b-moon{background:#1a0a2e;color:var(--purple);border:1px solid var(--purple);}
.log-box{background:var(--panel);border:1px solid var(--border);
    border-radius:10px;padding:12px 16px;font-size:12px;
    max-height:250px;overflow-y:auto;
    font-family:'JetBrains Mono',monospace;direction:ltr;}
.log-buy{color:var(--green);}
.log-sell{color:var(--red);}
.log-info{color:var(--accent);}
.log-warn{color:var(--yellow);}
.log-scan{color:#3a5a7a;}
.log-fail{color:#6a2a3a;}
.log-pass{color:#00cc7a;}
.log-moon{color:var(--purple);}
.log-withdraw{color:var(--yellow);}
.pulse{display:inline-block;width:9px;height:9px;border-radius:50%;
    background:var(--green);animation:blink 1.2s infinite;margin-left:6px;}
@keyframes blink{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.6)}}
.pulse.off{background:var(--red);animation:none;}
.prog-wrap{background:#0a1a2a;border-radius:6px;height:10px;overflow:hidden;margin-top:6px;}
.prog-bar{height:100%;border-radius:6px;
    background:linear-gradient(90deg,var(--accent),var(--green));transition:width .5s;}
.stButton>button{
    font-family:'Cairo',sans-serif!important;font-weight:700!important;
    font-size:14px!important;border-radius:8px!important;
    border:1px solid var(--accent)!important;
    color:var(--accent)!important;background:transparent!important;
    transition:all .2s!important;width:100%;}
.stButton>button:hover{background:var(--accent)!important;color:var(--bg)!important;}
.sb>button{border-color:var(--green)!important;color:var(--green)!important;}
.sb>button:hover{background:var(--green)!important;color:var(--bg)!important;}
.xb>button{border-color:var(--red)!important;color:var(--red)!important;}
.xb>button:hover{background:var(--red)!important;color:var(--bg)!important;}
.wb>button{border-color:var(--yellow)!important;color:var(--yellow)!important;}
.wb>button:hover{background:var(--yellow)!important;color:var(--bg)!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:10px!important;}
.stNumberInput input{
    background:var(--panel)!important;border:1px solid var(--border)!important;
    color:var(--text)!important;font-family:'JetBrains Mono',monospace!important;
    border-radius:6px!important;}
.withdraw-banner{
    background:linear-gradient(135deg,#2a1a00,#3a2a00);
    border:2px solid var(--yellow);border-radius:12px;
    padding:16px 20px;margin-bottom:12px;font-family:'Cairo',sans-serif;
    animation:glow 1s infinite alternate;}
@keyframes glow{
    from{box-shadow:0 0 5px var(--yellow);}
    to{box-shadow:0 0 20px var(--yellow),0 0 40px #ff8c0050;}}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  الثوابت
# ════════════════════════════════════════════════════════════
MAX_OPEN       = 3
MAX_PRICE      = 0.001
COMMISSION     = 0.002
INITIAL_SL_PCT = 0.012
TRAILING_DIST  = 0.007
MFI_THRESH     = 55
VOL_MULT       = 2.0
MIN_AI_PROB    = 0.60
BASE_TP        = 0.018
TRAILING_DIST  = 0.007
MOON_THRESHOLD = 0.82
REVERSAL_DROP  = 0.008
SCAN_DELAY     = 0.4

# ════════════════════════════════════════════════════════════
#  الاتصال بالبورصة
# ════════════════════════════════════════════════════════════
@st.cache_resource
def get_exchange():
    return ccxt.mexc({"enableRateLimit": True, "timeout": 15000})

exchange = get_exchange()

# ════════════════════════════════════════════════════════════
#  السجلات — تكتب في المتغير العالمي فقط
# ════════════════════════════════════════════════════════════
def add_log(msg, kind="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    with _lock:
        _state["logs"].appendleft({"ts": ts, "msg": msg, "kind": kind})

def add_scan_result(symbol, status, reason, ai_prob=0.0, price=0.0):
    with _lock:
        _state["scan_results"].appendleft({
            "symbol":  symbol,
            "status":  status,
            "reason":  reason,
            "ai_prob": ai_prob,
            "price":   price,
            "time":    datetime.now().strftime("%H:%M:%S"),
        })

def add_alert(msg):
    with _lock:
        _state["alerts"].appendleft({
            "msg":  msg,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

# ════════════════════════════════════════════════════════════
#  المؤشرات الفنية
# ════════════════════════════════════════════════════════════
def compute_mfi(highs, lows, closes, volumes, period=14):
    tps = (np.array(highs)+np.array(lows)+np.array(closes))/3
    mf  = tps * np.array(volumes)
    pos = np.zeros(len(tps))
    neg = np.zeros(len(tps))
    for i in range(1, len(tps)):
        (pos if tps[i] > tps[i-1] else neg)[i] = mf[i]
    vals = []
    for i in range(period, len(tps)):
        p = pos[i-period:i].sum()
        n = neg[i-period:i].sum()
        vals.append(100.0 if n == 0 else 100-100/(1+p/n))
    return vals[-1] if vals else 50.0

def compute_rsi(closes, period=14):
    if len(closes) < period+1: return 50.0
    d  = np.diff(closes)
    g  = np.where(d > 0, d, 0)
    l  = np.where(d < 0, -d, 0)
    ag = np.mean(g[-period:])
    al = np.mean(l[-period:])
    return 100.0 if al == 0 else 100-100/(1+ag/al)

def compute_momentum(closes, period=5):
    if len(closes) < period+1: return 0.0
    return (closes[-1]-closes[-period-1])/closes[-period-1]*100

def compute_atr(highs, lows, closes, period=14):
    trs = [max(highs[i]-lows[i],
               abs(highs[i]-closes[i-1]),
               abs(lows[i]-closes[i-1]))
           for i in range(1, len(closes))]
    return np.mean(trs[-period:]) if trs else closes[-1]*0.01

def compute_volume_ratio(volumes, lookback=20):
    if len(volumes) < lookback+1: return 1.0
    avg = np.mean(volumes[-lookback-1:-1])
    return 1.0 if avg == 0 else volumes[-1]/avg

def compute_ema(closes, period=20):
    if len(closes) < period: return closes[-1]
    k = 2/(period+1)
    e = closes[0]
    for c in closes[1:]: e = c*k+e*(1-k)
    return e

def detect_reversal(closes, trail_high, current_price):
    if trail_high <= 0: return False
    drop = (trail_high - current_price) / trail_high
    momentum = compute_momentum(closes[-6:], 3) if len(closes) >= 6 else 0
    return drop >= REVERSAL_DROP and momentum < -0.2

# ════════════════════════════════════════════════════════════
#  الذكاء الاصطناعي
# ════════════════════════════════════════════════════════════
def get_features(mfi, vol_ratio, momentum, atr_pct, rsi, ema_dist, vol_trend):
    return [mfi, vol_ratio, momentum, atr_pct, rsi, ema_dist, vol_trend]

def ai_predict(features):
    with _lock:
        m  = _state["ai_model"]
        sc = _state["ai_scaler"]
        trained = _state["ai_trained"]
    if not trained or m is None:
        mfi,vol_ratio,momentum,atr_pct,rsi,ema_dist,vol_trend = features
        s = 0.0
        if mfi > MFI_THRESH:      s += 0.25
        if vol_ratio > VOL_MULT:  s += 0.25
        if momentum > 0.3:        s += 0.15
        if rsi > 50 and rsi < 78: s += 0.15
        if ema_dist > 0:          s += 0.10
        if vol_trend > 1.5:       s += 0.10
        return min(s, 0.99)
    try:
        X = sc.transform([features])
        return float(m.predict_proba(X)[0][1])
    except:
        return 0.5

def ai_retrain():
    with _lock:
        hist = list(_state["ai_history"])
    if len(hist) < 20: return
    X = [h[0] for h in hist]
    y = [h[1] for h in hist]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=4,
        learning_rate=0.07, subsample=0.8,
        min_samples_leaf=3, random_state=42,
    )
    try:
        model.fit(Xs, y)
        with _lock:
            _state["ai_model"]   = model
            _state["ai_scaler"]  = sc
            _state["ai_trained"] = True
        add_log(f"🧠 تم تدريب الذكاء الاصطناعي على {len(hist)} صفقة", "info")
    except Exception as e:
        add_log(f"⚠ خطأ في تدريب AI: {e}", "warn")

def ai_record(features, won):
    with _lock:
        _state["ai_history"].append((features, int(won)))
        n = len(_state["ai_history"])
    if n % 10 == 0:
        ai_retrain()

# ════════════════════════════════════════════════════════════
#  جلب البيانات
# ════════════════════════════════════════════════════════════
def fetch_ohlcv(symbol, limit=60):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe="1m", limit=limit)
        if not data or len(data) < 20: return None
        return pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    except:
        return None

def fetch_ticker(symbol):
    try:
        return exchange.fetch_ticker(symbol)
    except:
        return None

def load_all_cheap_symbols():
    try:
        tickers = exchange.fetch_tickers()
        cheap = []
        for sym, t in tickers.items():
            if not sym.endswith("/USDT"): continue
            price = t.get("last") or 0
            vol   = t.get("quoteVolume") or 0
            if 0 < price <= MAX_PRICE and vol > 500:
                cheap.append({"symbol": sym, "vol": vol})
        cheap.sort(key=lambda x: x["vol"], reverse=True)
        result = [c["symbol"] for c in cheap]
        add_log(f"✓ تم تحميل {len(result)} عملة رخيصة", "info")
        return result
    except Exception as e:
        add_log(f"⚠ خطأ تحميل العملات: {e}", "warn")
        return []

# ════════════════════════════════════════════════════════════
#  منطق التداول
# ════════════════════════════════════════════════════════════
def check_compound():
    with _lock:
        bal  = _state["balance"]
        init = _state["initial_capital"]
        wdrn = _state["withdrawn"]
        ready = _state["withdraw_ready"]
    profit = (bal + wdrn) - init
    if profit >= init and not ready:
        with _lock:
            _state["withdraw_ready"] = True
            _state["compound_log"].append({
                "الوقت":    datetime.now().strftime("%Y-%m-%d %H:%M"),
                "الحدث":    "تضاعف رأس المال 🎉",
                "الرصيد":   round(bal, 4),
                "الربح":    round(profit, 4),
            })
        add_log(f"🎉 رأس المال تضاعف! الرصيد=${bal:.2f} الربح=${profit:.2f}", "withdraw")
        add_alert(f"💰 الربح وصل ${profit:.2f} — يمكنك سحب الأرباح!")

def open_trade(symbol, price, atr, features, is_moon=False):
    with _lock:
        open_trades = _state["open_trades"]
        balance     = _state["balance"]
        if len(open_trades) >= MAX_OPEN: return
        if balance < 1.0: return
        alloc    = balance / MAX_OPEN
        fee_in   = alloc * COMMISSION
        qty      = (alloc - fee_in) / price
        sl       = price * (1 - INITIAL_SL_PCT)
        tp       = price * (1 + (999 if is_moon else BASE_TP))
        trade = {
            "symbol":      symbol,
            "entry_price": price,
            "qty":         qty,
            "alloc":       alloc,
            "fee_in":      fee_in,
            "sl":          sl,
            "tp":          tp,
            "trail_high":  price,
            "trail_sl":    sl,
            "is_moon":     is_moon,
            "peak_gain":   0.0,
            "entry_time":  datetime.now().isoformat(timespec="seconds"),
            "exit_price":  None,
            "exit_time":   None,
            "pnl_pct":     None,
            "status":      "OPEN",
            "features":    features,
        }
        _state["open_trades"][symbol] = trade
        _state["balance"] -= alloc
        _state["stats"]["total_fees"] += fee_in

    mode = "🌙 وضع القمر" if is_moon else "⚡ عادي"
    add_log(f"▲ شراء {symbol} @ ${price:.8f} | {mode}", "buy")
    add_alert(f"✅ شراء: {symbol} [{mode}]")

def close_trade(symbol, price, reason=""):
    with _lock:
        trade = _state["open_trades"].pop(symbol, None)
    if not trade: return

    fee_out  = trade["qty"] * price * COMMISSION
    gross    = (price - trade["entry_price"]) * trade["qty"]
    net_usdt = gross - trade["fee_in"] - fee_out
    pnl_pct  = net_usdt / trade["alloc"] * 100

    with _lock:
        _state["balance"] += trade["qty"] * price - fee_out
        _state["stats"]["total_fees"] += fee_out
        _state["stats"]["pnl_today"]  += net_usdt
        _state["stats"]["pnl_total"]  += net_usdt

    trade.update({
        "exit_price": price,
        "exit_time":  datetime.now().isoformat(timespec="seconds"),
        "pnl_pct":    round(pnl_pct, 3),
        "pnl_net":    round(net_usdt, 6),
        "status":     "WIN" if pnl_pct > 0 else "LOSS",
    })

    with _lock:
        _state["trades"].append(trade)
        if pnl_pct > 0:
            _state["stats"]["win"] += 1
            _state["stats"]["best_trade"] = max(_state["stats"]["best_trade"], pnl_pct)
            if pnl_pct > 10:
                _state["stats"]["moon_shots"] += 1
        else:
            _state["stats"]["loss"] += 1
            _state["stats"]["worst_trade"] = min(_state["stats"]["worst_trade"], pnl_pct)

    if pnl_pct > 0:
        add_log(f"▼ بيع {symbol} | ربح: {pnl_pct:+.2f}% [{reason}]",
                "moon" if pnl_pct > 10 else "buy")
        add_alert(f"💰 ربح: {symbol} {pnl_pct:+.2f}%")
    else:
        add_log(f"▼ بيع {symbol} | خسارة: {pnl_pct:+.2f}% [{reason}]", "sell")
        add_alert(f"⛔ خسارة: {symbol} {pnl_pct:+.2f}%")

    ai_record(trade["features"], pnl_pct > 0)
    check_compound()

def manage_open_trades():
    with _lock:
        syms = list(_state["open_trades"].keys())
    for sym in syms:
        ticker = fetch_ticker(sym)
        if not ticker: continue
        price = ticker.get("last") or 0
        if not price: continue
        with _lock:
            t = _state["open_trades"].get(sym)
            if not t: continue
            if price > t["trail_high"]:
                t["trail_high"] = price
                t["trail_sl"]   = price * (1 - TRAILING_DIST)
            peak_gain = (t["trail_high"]-t["entry_price"])/t["entry_price"]*100
            t["peak_gain"]  = peak_gain
            is_moon = t["is_moon"]
            sl      = t["sl"]
            trail_sl = t["trail_sl"]
            tp      = t["tp"]
            trail_high = t["trail_high"]

        if is_moon:
            df = fetch_ohlcv(sym, limit=15)
            closes = df["close"].tolist() if df is not None else [price]
            if detect_reversal(closes, trail_high, price):
                close_trade(sym, price, f"ارتداد من القمة ({peak_gain:+.1f}%)")
            elif price <= sl:
                close_trade(sym, price, "وقف الخسارة")
        else:
            if price >= tp:
                close_trade(sym, price, "هدف الربح ✓")
            elif price <= trail_sl:
                close_trade(sym, price, "وقف متحرك")
            elif price <= sl:
                close_trade(sym, price, "وقف الخسارة")

def scan_symbols():
    with _lock:
        symbols = list(_state["all_symbols"])

    if not symbols: return
    results = []

    for sym in symbols:
        if _stop_event.is_set(): break

        with _lock:
            _state["current_scan"]   = sym
            _state["stats"]["total_scanned"] += 1

        try:
            with _lock:
                open_count = len(_state["open_trades"])
                open_syms  = list(_state["open_trades"].keys())
                balance    = _state["balance"]

            if open_count >= MAX_OPEN and sym not in open_syms:
                time.sleep(SCAN_DELAY * 0.1)
                continue

            ticker = fetch_ticker(sym)
            if not ticker:
                time.sleep(SCAN_DELAY)
                continue
            price = ticker.get("last") or 0
            if not price or price > MAX_PRICE:
                time.sleep(SCAN_DELAY * 0.1)
                continue

            df = fetch_ohlcv(sym)
            if df is None:
                time.sleep(SCAN_DELAY)
                continue

            highs   = df["high"].tolist()
            lows    = df["low"].tolist()
            closes  = df["close"].tolist()
            volumes = df["volume"].tolist()

            mfi       = compute_mfi(highs, lows, closes, volumes)
            vol_ratio = compute_volume_ratio(volumes)
            momentum  = compute_momentum(closes)
            atr       = compute_atr(highs, lows, closes)
            atr_pct   = atr / closes[-1] if closes[-1] > 0 else 0.01
            rsi       = compute_rsi(closes)
            ema20     = compute_ema(closes, 20)
            ema_dist  = (closes[-1]-ema20)/ema20*100
            vol_trend = volumes[-1]/volumes[-2] if len(volumes)>1 and volumes[-2]>0 else 1.0

            features = get_features(mfi, vol_ratio, momentum, atr_pct, rsi, ema_dist, vol_trend)
            ai_prob  = ai_predict(features)
            is_moon  = ai_prob >= MOON_THRESHOLD and vol_ratio > 3.0 and mfi > 65

            if is_moon:            signal = "🌙 قمر"
            elif ai_prob > 0.70:   signal = "⚡ قوي"
            elif ai_prob > 0.60:   signal = "◎ جيد"
            elif ai_prob > 0.45:   signal = "○ متابعة"
            else:                  signal = "✗ ضعيف"

            results.append({
                "symbol": sym, "price": price,
                "mfi": round(mfi,1), "vol_ratio": round(vol_ratio,2),
                "momentum": round(momentum,3), "ai_prob": round(ai_prob*100,1),
                "signal": signal, "is_moon": is_moon,
                "features": features, "atr": atr,
            })

            can_enter = (
                sym not in open_syms
                and open_count < MAX_OPEN
                and balance > 1.0
                and mfi > MFI_THRESH
                and vol_ratio > VOL_MULT
                and ai_prob >= MIN_AI_PROB
            )

            if can_enter:
                add_scan_result(sym, "moon" if is_moon else "pass",
                    f"دخول! AI={ai_prob*100:.0f}% MFI={mfi:.0f}", ai_prob, price)
                open_trade(sym, price, atr, features, is_moon)
            else:
                if mfi <= MFI_THRESH:         reason = f"MFI منخفض ({mfi:.0f})"
                elif vol_ratio <= VOL_MULT:   reason = f"حجم ضعيف (×{vol_ratio:.1f})"
                elif ai_prob < MIN_AI_PROB:   reason = f"AI منخفض ({ai_prob*100:.0f}%)"
                else:                          reason = "صفقات مكتملة أو رصيد"
                add_scan_result(sym, "fail", reason, ai_prob, price)

        except Exception as e:
            add_log(f"⚠ {sym}: {str(e)[:40]}", "warn")

        time.sleep(SCAN_DELAY)

    results.sort(key=lambda x: x["ai_prob"], reverse=True)
    with _lock:
        _state["scan_list"]    = results[:30]
        _state["last_update"]  = datetime.now().strftime("%H:%M:%S")
        _state["current_scan"] = ""
        _state["scan_cycle"]  += 1

# ════════════════════════════════════════════════════════════
#  خيط البوت — يعمل 24/7 بدون session_state
# ════════════════════════════════════════════════════════════
def bot_loop():
    add_log("⚡ البوت يعمل 24/7 — جاري تحميل العملات...", "info")
    symbols = load_all_cheap_symbols()
    with _lock:
        _state["all_symbols"]    = symbols
        _state["symbols_loaded"] = True
    add_log(f"📋 {len(symbols)} عملة جاهزة للفحص", "info")

    while not _stop_event.is_set():
        try:
            manage_open_trades()
            scan_symbols()
            # تحديث قائمة العملات كل ساعة تقريباً
            with _lock:
                cycle = _state["scan_cycle"]
            if cycle % 60 == 0 and cycle > 0:
                new_syms = load_all_cheap_symbols()
                with _lock:
                    _state["all_symbols"] = new_syms
        except Exception as e:
            add_log(f"⚠ خطأ عام: {str(e)[:60]}", "warn")
            time.sleep(10)

    add_log("■ توقف البوت.", "warn")

# ════════════════════════════════════════════════════════════
#  تشغيل/إيقاف
# ════════════════════════════════════════════════════════════
def start_bot(budget):
    if _state["running"]: return
    _stop_event.clear()
    with _lock:
        _state["running"]         = True
        _state["balance"]         = budget
        _state["initial_capital"] = budget
        _state["withdrawn"]       = 0.0
        _state["withdraw_ready"]  = False
        _state["trades"]          = []
        _state["open_trades"]     = {}
        _state["scan_results"]    = deque(maxlen=50)
        _state["logs"]            = deque(maxlen=120)
        _state["alerts"]          = deque(maxlen=5)
        _state["scan_cycle"]      = 0
        _state["scan_list"]       = []
        _state["all_symbols"]     = []
        _state["stats"] = {
            "win":0,"loss":0,"pnl_today":0.0,"pnl_total":0.0,
            "total_fees":0.0,"best_trade":0.0,"worst_trade":0.0,
            "moon_shots":0,"total_scanned":0,
        }
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()

def stop_bot():
    _stop_event.set()
    with _lock:
        _state["running"] = False

def do_withdraw():
    with _lock:
        bal  = _state["balance"]
        init = _state["initial_capital"]
        if bal <= init: return
        profit = bal - init
        _state["withdrawn"] += profit
        _state["balance"]    = init
        _state["withdraw_ready"] = False
    add_log(f"💸 تم سحب ${profit:.4f} — الرصيد المتداول: ${init:.2f}", "withdraw")
    add_alert(f"✅ تم سحب: ${profit:.4f}")

# ════════════════════════════════════════════════════════════
#  قراءة الحالة للعرض (آمنة)
# ════════════════════════════════════════════════════════════
def snap():
    """لقطة آمنة من الحالة للعرض في الواجهة"""
    with _lock:
        import copy
        return copy.deepcopy(_state)

# ════════════════════════════════════════════════════════════
#  الشريط الجانبي
# ════════════════════════════════════════════════════════════
# حقل الميزانية في session_state فقط (للواجهة)
if "budget_ui" not in st.session_state:
    st.session_state.budget_ui = 20.0

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 16px;border-bottom:1px solid #1a3050;margin-bottom:16px;">
    <div style="font-family:Cairo;font-size:20px;font-weight:900;color:#00c8ff;">⚡ بوت المضاربة</div>
    <div style="font-size:11px;color:#4a6a8a;">MEXC Scalping Bot v3.1</div>
    </div>""", unsafe_allow_html=True)

    running = _state["running"]
    st.markdown(
        f'<div style="text-align:center;margin-bottom:14px;">'
        f'<span style="font-size:15px;font-weight:700;">'
        f'{"🟢 يعمل 24/7" if running else "🔴 متوقف"}</span></div>',
        unsafe_allow_html=True)

    budget = st.number_input("💰 الميزانية (USDT)",
        min_value=5.0, max_value=10000.0,
        value=st.session_state.budget_ui, step=5.0,
        disabled=running)
    st.session_state.budget_ui = budget

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sb">', unsafe_allow_html=True)
        if st.button("▶ تشغيل", use_container_width=True, disabled=running):
            start_bot(budget)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="xb">', unsafe_allow_html=True)
        if st.button("■ إيقاف", use_container_width=True, disabled=not running):
            stop_bot()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    s = snap()
    if s["withdraw_ready"]:
        st.markdown('<div class="wb">', unsafe_allow_html=True)
        if st.button("💸 سحب الأرباح", use_container_width=True):
            do_withdraw(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    bal  = s["balance"]
    init = s["initial_capital"]
    wdrn = s["withdrawn"]
    pct  = (bal-init)/init*100 if init > 0 else 0
    col  = "#00ff9d" if bal >= init else "#ff3860"
    prog = min((bal+wdrn-init)/init*100, 100) if init > 0 else 0

    st.markdown(
        f'<div class="ml">الرصيد المتداول</div>'
        f'<div style="font-size:22px;font-weight:900;color:{col};font-family:JetBrains Mono,monospace;">${bal:.4f}</div>'
        f'<div style="font-size:11px;color:{col};margin-bottom:6px;">{pct:+.2f}%</div>'
        f'<div class="ml">تقدم المضاعفة ({prog:.1f}%)</div>'
        f'<div class="prog-wrap"><div class="prog-bar" style="width:{prog}%"></div></div>',
        unsafe_allow_html=True)

    if wdrn > 0:
        st.markdown(
            f'<div style="margin-top:8px;" class="ml">إجمالي المسحوب</div>'
            f'<div style="font-size:16px;font-weight:700;color:#ffd600;font-family:JetBrains Mono,monospace;">${wdrn:.4f}</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    n_sym = len(s["all_symbols"])
    st.markdown(f"""
    <div style="font-size:11px;color:#3a6a8a;line-height:2.2;">
    🔍 العملات المفحوصة: <b style="color:#00c8ff;">{n_sym}</b><br>
    🌙 وضع القمر: AI > 82%<br>
    🎯 TP عادي: 1.8% | قمر: ∞<br>
    🛡 SL: 1.2% | Trailing: 0.7%<br>
    💸 عمولة: 0.2% × 2<br>
    🔄 Compound تلقائي<br>
    ⏰ يعمل 24/7 بلا توقف
    </div>""", unsafe_allow_html=True)

    if st.button("🔄 تحديث الواجهة", use_container_width=True):
        st.rerun()

# ════════════════════════════════════════════════════════════
#  لوحة التحكم الرئيسية
# ════════════════════════════════════════════════════════════
s = snap()

st.markdown(
    '<h1 style="font-family:Cairo,sans-serif;font-size:24px;font-weight:900;'
    'color:#00c8ff;margin:0 0 2px;">🚀 مركز التحكم — بوت المضاربة v3.1</h1>'
    '<div style="font-size:11px;color:#3a6a8a;margin-bottom:14px;">'
    '⚡ إصلاح threading | فحص شامل | Compound تلقائي | 24/7</div>',
    unsafe_allow_html=True)

# تنبيه المضاعفة
if s["withdraw_ready"]:
    bal2  = s["balance"]
    init2 = s["initial_capital"]
    prof2 = bal2 - init2
    st.markdown(
        f'<div class="withdraw-banner">'
        f'<div style="font-size:16px;font-weight:900;color:#ffd600;">🎉 رأس المال تضاعف!</div>'
        f'<div style="font-size:13px;margin-top:6px;">البداية: ${init2:.2f} | الرصيد: ${bal2:.2f} | الربح: ${prof2:.2f}</div>'
        f'</div>', unsafe_allow_html=True)

# تنبيهات
for a in list(s["alerts"])[:2]:
    st.markdown(
        f'<div style="background:#002a1a;border:1px solid #00ff9d;border-radius:8px;'
        f'padding:10px 16px;margin-bottom:6px;font-size:13px;color:#00ff9d;">'
        f'🔔 [{a["time"]}] {a["msg"]}</div>', unsafe_allow_html=True)

# المقاييس
st2   = s["stats"]
total = st2["win"] + st2["loss"]
wr    = st2["win"]/total*100 if total > 0 else 0

def mc(col, label, val, sub="", css=""):
    with col:
        st.markdown(
            f'<div class="mc {css}"><div class="ml">{label}</div>'
            f'<div class="mv">{val}</div><div class="ms">{sub}</div></div>',
            unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
mc(c1,"ربح اليوم",f'{st2["pnl_today"]:+.3f}$',"صافٍ","g" if st2["pnl_today"]>=0 else "r")
mc(c2,"الربح الكلي",f'{st2["pnl_total"]:+.3f}$',"منذ البداية","g" if st2["pnl_total"]>=0 else "r")
mc(c3,"ربح/خسارة",f'{st2["win"]}/{st2["loss"]}',f'{total} صفقة',"y")
mc(c4,"نسبة النجاح",f'{wr:.1f}%',"AI فعّال ✓" if s["ai_trained"] else "يتعلم...","g" if wr>=50 else "r")
mc(c5,"مفتوحة",f'{len(s["open_trades"])}/{MAX_OPEN}',"حالياً","o")
mc(c6,"🌙 القمر",f'{st2["moon_shots"]}',"صفقات كبيرة","p")
mc(c7,"مفحوصة",f'{st2["total_scanned"]:,}',"إجمالي","")

# الفحص المباشر
st.markdown('<div class="sh">🔍 الفحص المباشر</div>', unsafe_allow_html=True)
cur = s["current_scan"]
cyc = s["scan_cycle"]
ca, cb = st.columns([3,1])
with ca:
    if cur:
        st.markdown(
            f'<div class="sc a"><div><span class="sym">⟳ {cur}</span>'
            f'<div class="rsn">جارٍ التحليل...</div></div>'
            f'<span class="badge b-scan">يفحص</span></div>',
            unsafe_allow_html=True)
    else:
        msg = "في انتظار الدورة التالية..." if s["running"] else "شغّل البوت للبدء"
        st.markdown(f'<div class="sc" style="justify-content:center;color:#3a6a8a;">{msg}</div>',
                    unsafe_allow_html=True)
with cb:
    st.markdown(
        f'<div class="mc"><div class="ml">دورة الفحص</div>'
        f'<div class="mv" style="font-size:18px;">#{cyc}</div>'
        f'<div class="ms">{len(s["all_symbols"])} عملة</div></div>',
        unsafe_allow_html=True)

for r in list(s["scan_results"])[:10]:
    is_m = r["status"] == "moon"
    cc   = "moon" if is_m else ("p" if r["status"]=="pass" else "f")
    bc   = "b-moon" if is_m else ("b-pass" if r["status"]=="pass" else "b-fail")
    bt   = "🌙 قمر" if is_m else ("✓ دخلنا" if r["status"]=="pass" else "✗ رُفض")
    bw   = int(r["ai_prob"]*55)
    st.markdown(
        f'<div class="sc {cc}">'
        f'<div style="flex:1;"><span class="sym">{r["symbol"]}</span>'
        f'<div class="rsn">{r["reason"]}</div>'
        f'<div style="display:flex;align-items:center;gap:6px;margin-top:3px;">'
        f'<div style="width:{bw}px;height:4px;border-radius:2px;background:linear-gradient(90deg,#00c8ff,#00ff9d);"></div>'
        f'<span style="font-size:10px;color:#4a6a8a;">{r["ai_prob"]*100:.0f}%</span></div></div>'
        f'<div style="text-align:left;"><span class="badge {bc}">{bt}</span>'
        f'<div style="font-size:10px;color:#3a6a8a;margin-top:3px;">{r["time"]}</div></div></div>',
        unsafe_allow_html=True)

# الصفقات المفتوحة
st.markdown('<div class="sh">🔴 الصفقات المفتوحة</div>', unsafe_allow_html=True)
if s["open_trades"]:
    rows = []
    for sym, t in s["open_trades"].items():
        tk  = fetch_ticker(sym)
        cur_p = tk["last"] if tk else t["entry_price"]
        net = ((cur_p-t["entry_price"])*t["qty"] - t["fee_in"] - t["qty"]*cur_p*COMMISSION)
        pnl = net/t["alloc"]*100
        rows.append({
            "العملة":     sym,
            "النوع":      "🌙" if t["is_moon"] else "⚡",
            "الدخول $":  f'{t["entry_price"]:.8f}',
            "الحالي $":  f'{cur_p:.8f}',
            "أعلى ربح":  f'{t["peak_gain"]:+.2f}%',
            "وقف متحرك": f'{t["trail_sl"]:.8f}',
            "PnL %":     f'{pnl:+.3f}%',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.markdown('<div style="text-align:center;padding:16px;color:#3a6a8a;">⊘ لا صفقات مفتوحة</div>',
                unsafe_allow_html=True)

# أفضل الإشارات
st.markdown('<div class="sh">📡 أعلى إشارات AI</div>', unsafe_allow_html=True)
if s["scan_list"]:
    rows = [{"العملة":x["symbol"],"السعر $":f'{x["price"]:.8f}',
             "MFI":x["mfi"],"حجم ×":f'×{x["vol_ratio"]}',
             "زخم":f'{x["momentum"]:+.3f}%',
             "AI %":f'{x["ai_prob"]}%',"الإشارة":x["signal"]}
            for x in s["scan_list"][:10]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# سجل الصفقات
st.markdown('<div class="sh">📋 سجل الصفقات</div>', unsafe_allow_html=True)
if s["trades"]:
    rows = []
    for t in reversed(s["trades"][-60:]):
        pnl = t.get("pnl_pct",0) or 0
        net = t.get("pnl_net",0) or 0
        rows.append({
            "العملة":   t["symbol"],
            "النوع":    "🌙" if t.get("is_moon") else "⚡",
            "الدخول":  t["entry_time"],
            "دخول $":  f'{t["entry_price"]:.8f}',
            "خروج $":  f'{t["exit_price"]:.8f}' if t["exit_price"] else "—",
            "PnL %":   f'{pnl:+.3f}%',
            "صافٍ $":  f'{net:+.6f}',
            "الحالة":   "🟢 ربح" if t["status"]=="WIN" else "🔴 خسارة",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# سجل الأحداث
st.markdown('<div class="sh">📟 سجل الأحداث</div>', unsafe_allow_html=True)
filt = st.selectbox("عرض:",
    ["الكل","الصفقات","نتائج الفحص","التحذيرات","القمر 🌙"],
    label_visibility="collapsed")
fmap = {
    "الكل":        None,
    "الصفقات":     ["buy","sell","moon","withdraw"],
    "نتائج الفحص": ["pass","fail","scan"],
    "التحذيرات":   ["warn"],
    "القمر 🌙":    ["moon"],
}
chosen = fmap[filt]
logs_all = list(s["logs"])
logs_show = [l for l in logs_all if chosen is None or l["kind"] in chosen]
if logs_show:
    html = '<div class="log-box">'
    for e in logs_show[:60]:
        html += f'<div class="log-{e["kind"]}">[{e["ts"]}] {e["msg"]}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
else:
    st.markdown('<div class="log-box" style="color:#3a6a8a;text-align:center;">في انتظار الأحداث...</div>',
                unsafe_allow_html=True)

# حالة AI
st.markdown('<div class="sh">🧠 حالة الذكاء الاصطناعي</div>', unsafe_allow_html=True)
a1,a2,a3,a4 = st.columns(4)
trained = s["ai_trained"]
n_hist  = len(s["ai_history"])
for col,label,val,sub,css in [
    (a1,"الحالة","✓ مُدرَّب" if trained else "⟳ يتعلم","Gradient Boosting","g" if trained else "y"),
    (a2,"العينات",str(n_hist),f"تقدم: {min(n_hist/20*100,100):.0f}%",""),
    (a3,"المؤشرات","7","MFI،RSI،EMA،ATR،حجم،زخم،اتجاه",""),
    (a4,"عتبة القمر 🌙","82%","لا سقف للربح","p"),
]:
    with col:
        st.markdown(
            f'<div class="mc {css}"><div class="ml">{label}</div>'
            f'<div class="mv" style="font-size:16px;">{val}</div>'
            f'<div class="ms">{sub}</div></div>',
            unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:10px;color:#1a3050;">'
    '⚡ بوت المضاربة v3.1 — وضع التداول الوهمي — ليس نصيحة مالية ⚡</div>',
    unsafe_allow_html=True)

# تحديث تلقائي
if _state["running"]:
    time.sleep(8)
    st.rerun()
