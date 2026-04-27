"""
══════════════════════════╗
║   MEXC SCALPING BOT — Paper Trading Mode                ║
║   Strategy: Volume Breakout + MFI + AI (GradientBoost)  ║
║   Platform: Streamlit | Data: CCXT | AI: Scikit-learn   ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
import threading
import json
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MEXC Scalping Bot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  GLOBAL CSS — DARK TERMINAL THEME
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
    --bg-deep:    #080d12;
    --bg-card:    #0d1520;
    --bg-panel:   #111c2a;
    --border:     #1e3050;
    --accent:     #00d4ff;
    --green:      #00ff9d;
    --red:        #ff3860;
    --yellow:     #ffd600;
    --muted:      #4a6280;
    --text:       #c8ddf0;
    --text-dim:   #5a7a99;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
.block-container { padding: 1rem 1.5rem !important; }

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
}
.metric-card.green::before { background: var(--green); }
.metric-card.red::before   { background: var(--red); }
.metric-card.yellow::before { background: var(--yellow); }
.metric-label {
    font-size: 10px; letter-spacing: 2px;
    color: var(--text-dim); text-transform: uppercase;
    margin-bottom: 6px; font-family: 'Rajdhani', sans-serif;
}
.metric-value {
    font-size: 26px; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.metric-sub { font-size: 11px; color: var(--text-dim); margin-top: 4px; }

.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 13px; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase;
    color: var(--accent); border-bottom: 1px solid var(--border);
    padding-bottom: 8px; margin: 20px 0 12px;
}

.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 2px !important;
    font-size: 14px !important; border-radius: 6px !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    background: transparent !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg-deep) !important;
}
.start-btn > button {
    border-color: var(--green) !important; color: var(--green) !important;
}
.start-btn > button:hover {
    background: var(--green) !important; color: var(--bg-deep) !important;
}
.stop-btn > button {
    border-color: var(--red) !important; color: var(--red) !important;
}
.stop-btn > button:hover {
    background: var(--red) !important; color: var(--bg-deep) !important;
}

.stNumberInput input, .stTextInput input {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px !important;
}

.log-box {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12px;
    max-height: 200px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
}
.log-buy  { color: var(--green); }
.log-sell { color: var(--red); }
.log-info { color: var(--accent); }
.log-warn { color: var(--yellow); }

.pulse {
    display: inline-block;
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    animation: pulse 1.2s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: .4; transform: scale(.7); }
}
.pulse.stopped { background: var(--red); animation: none; }

.sidebar-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 18px; font-weight: 700;
    color: var(--accent); letter-spacing: 2px;
    text-align: center; margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "running": False,
        "budget": 100.0,
        "balance": 100.0,
        "trades": [],
        "scan_list": [],
        "open_trades": {},
        "logs": deque(maxlen=60),
        "ai_model": None,
        "ai_scaler": None,
        "ai_trained": False,
        "ai_history": [],
        "thread": None,
        "stop_event": threading.Event(),
        "stats": {"win": 0, "loss": 0, "pnl_today": 0.0, "pnl_hour": 0.0},
        "last_update": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════
MAX_OPEN        = 2
TAKE_PROFIT_PCT = 0.015
INITIAL_SL_PCT  = 0.012
TRAILING_DIST   = 0.008
MFI_BUY_THRESH  = 55
VOL_MULTIPLIER  = 2.0
SCAN_INTERVAL   = 8
MAX_PRICE       = 0.001

WATCHLIST_BASE = [
    "PEPE/USDT", "SHIB/USDT", "FLOKI/USDT", "BONK/USDT",
    "WIF/USDT",  "LUNC/USDT", "XEC/USDT",   "HOT/USDT",
    "BTT/USDT",  "WIN/USDT",  "TRX/USDT",   "DOGE/USDT",
    "VOLT/USDT", "SAMO/USDT", "BABYDOGE/USDT",
]

# ════════════════════════════════════════════════════════════
#  EXCHANGE
# ════════════════════════════════════════════════════════════
@st.cache_resource
def get_exchange():
    return ccxt.mexc({"enableRateLimit": True})

exchange = get_exchange()

# ════════════════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════════════════
def compute_mfi(highs, lows, closes, volumes, period=14):
    tps = (np.array(highs) + np.array(lows) + np.array(closes)) / 3
    raw_mf = tps * np.array(volumes)
    pos, neg = np.zeros(len(tps)), np.zeros(len(tps))
    for i in range(1, len(tps)):
        if tps[i] > tps[i - 1]:
            pos[i] = raw_mf[i]
        else:
            neg[i] = raw_mf[i]
    mfi_vals = []
    for i in range(period, len(tps)):
        p = pos[i - period:i].sum()
        n = neg[i - period:i].sum()
        mfi_vals.append(100.0 if n == 0 else 100 - 100 / (1 + p / n))
    return mfi_vals[-1] if mfi_vals else 50.0

def compute_momentum(closes, period=5):
    if len(closes) < period + 1:
        return 0.0
    return (closes[-1] - closes[-period - 1]) / closes[-period - 1] * 100

def compute_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    return np.mean(trs[-period:]) if trs else closes[-1] * 0.01

def compute_volume_ratio(volumes, lookback=20):
    if len(volumes) < lookback + 1:
        return 1.0
    avg = np.mean(volumes[-lookback - 1:-1])
    return 1.0 if avg == 0 else volumes[-1] / avg

# ════════════════════════════════════════════════════════════
#  AI MODEL
# ════════════════════════════════════════════════════════════
def get_features(mfi, vol_ratio, momentum, atr_pct):
    return [mfi, vol_ratio, momentum, atr_pct]

def ai_predict(features):
    m  = st.session_state.ai_model
    sc = st.session_state.ai_scaler
    if m is None or not st.session_state.ai_trained:
        mfi, vol_ratio, momentum, atr_pct = features
        score = 0.0
        if mfi > MFI_BUY_THRESH:       score += 0.35
        if vol_ratio > VOL_MULTIPLIER:  score += 0.35
        if momentum > 0.3:              score += 0.20
        if atr_pct < 0.05:             score += 0.10
        return min(score, 0.99)
    try:
        X = sc.transform([features])
        return float(m.predict_proba(X)[0][1])
    except Exception:
        return 0.5

def ai_retrain():
    hist = st.session_state.ai_history
    if len(hist) < 20:
        return
    X = [h[0] for h in hist]
    y = [h[1] for h in hist]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3,
        learning_rate=0.1, subsample=0.8, random_state=42,
    )
    try:
        model.fit(Xs, y)
        st.session_state.ai_model   = model
        st.session_state.ai_scaler  = sc
        st.session_state.ai_trained = True
    except Exception:
        pass

def ai_record(features, won: bool):
    st.session_state.ai_history.append((features, int(won)))
    if len(st.session_state.ai_history) % 10 == 0:
        ai_retrain()

# ════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════
def add_log(msg: str, kind: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.appendleft({"ts": ts, "msg": msg, "kind": kind})

# ════════════════════════════════════════════════════════════
#  MARKET DATA
# ════════════════════════════════════════════════════════════
def fetch_ohlcv(symbol, timeframe="1m", limit=60):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not data or len(data) < 20:
            return None
        return pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    except Exception:
        return None

def fetch_ticker(symbol):
    try:
        return exchange.fetch_ticker(symbol)
    except Exception:
        return None

# ════════════════════════════════════════════════════════════
#  TRADE LOGIC
# ════════════════════════════════════════════════════════════
def open_trade(symbol, price, atr, features):
    slots_left = MAX_OPEN - len(st.session_state.open_trades)
    if slots_left <= 0:
        return
    alloc = st.session_state.balance / slots_left
    qty   = alloc / price
    sl    = price * (1 - INITIAL_SL_PCT)
    tp    = price * (1 + TAKE_PROFIT_PCT)
    trade = {
        "symbol":      symbol,
        "entry_price": price,
        "qty":         qty,
        "sl":          sl,
        "tp":          tp,
        "trail_high":  price,
        "trail_sl":    sl,
        "entry_time":  datetime.now().isoformat(timespec="seconds"),
        "exit_price":  None,
        "exit_time":   None,
        "pnl_pct":     None,
        "status":      "OPEN",
        "features":    features,
        "atr":         atr,
    }
    st.session_state.open_trades[symbol] = trade
    st.session_state.balance -= alloc
    add_log(f"BUY  {symbol} @ ${price:.8f}  SL=${sl:.8f}  TP=${tp:.8f}", "buy")

def close_trade(symbol, price, reason=""):
    trade = st.session_state.open_trades.pop(symbol, None)
    if not trade:
        return
    pnl_pct  = (price - trade["entry_price"]) / trade["entry_price"] * 100
    pnl_usdt = (price - trade["entry_price"]) * trade["qty"]
    st.session_state.balance += trade["qty"] * price
    trade.update({
        "exit_price": price,
        "exit_time":  datetime.now().isoformat(timespec="seconds"),
        "pnl_pct":    round(pnl_pct, 3),
        "status":     "WIN" if pnl_pct > 0 else "LOSS",
    })
    st.session_state.trades.append(trade)
    st.session_state.stats["pnl_today"] = round(
        st.session_state.stats["pnl_today"] + pnl_usdt, 4)
    st.session_state.stats["pnl_hour"]  = round(
        st.session_state.stats["pnl_hour"]  + pnl_usdt, 4)
    if pnl_pct > 0:
        st.session_state.stats["win"] += 1
        add_log(f"SELL {symbol} @ ${price:.8f}  PnL={pnl_pct:+.2f}%  [{reason}]", "buy")
    else:
        st.session_state.stats["loss"] += 1
        add_log(f"SELL {symbol} @ ${price:.8f}  PnL={pnl_pct:+.2f}%  [{reason}]", "sell")
    ai_record(trade["features"], pnl_pct > 0)

def manage_open_trades():
    for sym in list(st.session_state.open_trades.keys()):
        t      = st.session_state.open_trades[sym]
        ticker = fetch_ticker(sym)
        if not ticker:
            continue
        price = ticker["last"]
        if price > t["trail_high"]:
            t["trail_high"] = price
            t["trail_sl"]   = price * (1 - TRAILING_DIST)
        if price >= t["tp"]:
            close_trade(sym, price, "TP")
        elif price <= t["trail_sl"]:
            close_trade(sym, price, "Trail-SL")
        elif price <= t["sl"]:
            close_trade(sym, price, "SL")

def scan_symbols():
    results = []
    for sym in WATCHLIST_BASE:
        try:
            ticker = fetch_ticker(sym)
            if not ticker:
                continue
            price = ticker["last"]
            if price is None or price > MAX_PRICE:
                continue
            df = fetch_ohlcv(sym)
            if df is None:
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
            features  = get_features(mfi, vol_ratio, momentum, atr_pct)
            ai_prob   = ai_predict(features)

            results.append({
                "symbol":    sym,
                "price":     price,
                "mfi":       round(mfi, 1),
                "vol_ratio": round(vol_ratio, 2),
                "momentum":  round(momentum, 3),
                "ai_prob":   round(ai_prob * 100, 1),
                "signal":    "⚡ STRONG" if ai_prob > 0.65 else ("◎ WATCH" if ai_prob > 0.45 else "○ WEAK"),
                "features":  features,
                "atr":       atr,
            })

            # ── Entry decision ──
            if (sym not in st.session_state.open_trades
                    and len(st.session_state.open_trades) < MAX_OPEN
                    and mfi > MFI_BUY_THRESH
                    and vol_ratio > VOL_MULTIPLIER
                    and ai_prob > 0.60
                    and st.session_state.balance > 1.0):
                open_trade(sym, price, atr, features)

        except Exception:
            continue

    results.sort(key=lambda x: x["ai_prob"], reverse=True)
    st.session_state.scan_list  = results
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")

# ════════════════════════════════════════════════════════════
#  BOT THREAD
# ════════════════════════════════════════════════════════════
def bot_loop(stop_event: threading.Event):
    add_log("Bot started — Paper Trading mode", "info")
    while not stop_event.is_set():
        try:
            manage_open_trades()
            scan_symbols()
        except Exception as e:
            add_log(f"Error: {e}", "warn")
        for _ in range(SCAN_INTERVAL * 10):
            if stop_event.is_set():
                break
            time.sleep(0.1)
    add_log("Bot stopped.", "warn")

def start_bot():
    if st.session_state.running:
        return
    st.session_state.stop_event.clear()
    st.session_state.balance     = st.session_state.budget
    st.session_state.trades      = []
    st.session_state.open_trades = {}
    st.session_state.stats       = {"win": 0, "loss": 0, "pnl_today": 0.0, "pnl_hour": 0.0}
    st.session_state.logs        = deque(maxlen=60)
    t = threading.Thread(
        target=bot_loop,
        args=(st.session_state.stop_event,),
        daemon=True,
    )
    t.start()
    st.session_state.thread  = t
    st.session_state.running = True

def stop_bot():
    if not st.session_state.running:
        return
    st.session_state.stop_event.set()
    st.session_state.running = False

# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ SCALP BOT</div>', unsafe_allow_html=True)
    st.markdown("---")

    running  = st.session_state.running
    dot_cls  = "pulse" if running else "pulse stopped"
    stat_txt = "RUNNING" if running else "STOPPED"
    st.markdown(
        f'<div style="text-align:center;margin-bottom:16px;">'
        f'<span class="{dot_cls}"></span>'
        f'<span style="font-family:Rajdhani,sans-serif;font-size:14px;letter-spacing:2px;">{stat_txt}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    budget = st.number_input(
        "Budget (USDT)", min_value=10.0, max_value=10000.0,
        value=float(st.session_state.budget), step=10.0,
    )
    st.session_state.budget = budget

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        if st.button("▶ START", use_container_width=True):
            start_bot(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        if st.button("■ STOP", use_container_width=True):
            stop_bot(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#5a7a99;line-height:1.9;">
    📊 Volume Breakout + MFI<br>
    🤖 Gradient Boosting AI<br>
    🎯 TP: 1.5% | Init SL: 1.2%<br>
    🔄 Trailing Stop: 0.8%<br>
    💰 Max Price: $0.001<br>
    ⚡ Max Open: 2 Trades
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    bal     = st.session_state.balance
    ini     = st.session_state.budget
    bal_pct = (bal - ini) / ini * 100 if ini > 0 else 0
    col     = "#00ff9d" if bal >= ini else "#ff3860"
    st.markdown(
        f'<div class="metric-label">Current Balance</div>'
        f'<div style="font-size:22px;font-weight:700;color:{col};">${bal:.2f}</div>'
        f'<div style="font-size:11px;color:{col};">{bal_pct:+.2f}% from start</div>',
        unsafe_allow_html=True,
    )
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# ════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ════════════════════════════════════════════════════════════
st.markdown(
    '<h1 style="font-family:Rajdhani,sans-serif;font-size:28px;font-weight:700;'
    'color:#00d4ff;letter-spacing:4px;margin:0 0 4px;">MEXC SCALPING CONTROL CENTER</h1>'
    '<div style="font-size:11px;color:#4a6280;letter-spacing:2px;margin-bottom:20px;">'
    '⚡ PAPER TRADING MODE — LIVE MEXC PRICES</div>',
    unsafe_allow_html=True,
)

# ── TOP METRICS ──
stats = st.session_state.stats
total = stats["win"] + stats["loss"]
wr    = stats["win"] / total * 100 if total > 0 else 0.0
pnl_d = stats["pnl_today"]
pnl_h = stats["pnl_hour"]

def metric_card(col, label, value, sub="", css_class=""):
    with col:
        st.markdown(
            f'<div class="metric-card {css_class}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

c1, c2, c3, c4, c5 = st.columns(5)
metric_card(c1, "PnL Today",
            f'{"+" if pnl_d >= 0 else ""}{pnl_d:.2f} USDT',
            "Paper trading", "green" if pnl_d >= 0 else "red")
metric_card(c2, "PnL This Hour",
            f'{"+" if pnl_h >= 0 else ""}{pnl_h:.2f} USDT',
            "Rolling", "green" if pnl_h >= 0 else "red")
metric_card(c3, "Win / Loss",
            f'{stats["win"]} / {stats["loss"]}',
            f'{total} total trades', "yellow")
metric_card(c4, "Win Rate",
            f'{wr:.1f}%',
            "AI learning..." if not st.session_state.ai_trained else "AI active",
            "green" if wr >= 50 else "red")
metric_card(c5, "Open Trades",
            f'{len(st.session_state.open_trades)} / {MAX_OPEN}',
            "max 2 simultaneous", "")

# ── OPEN POSITIONS ──
st.markdown('<div class="section-header">🔴 LIVE OPEN POSITIONS</div>', unsafe_allow_html=True)
if st.session_state.open_trades:
    rows = []
    for sym, t in st.session_state.open_trades.items():
        ticker = fetch_ticker(sym)
        cur    = ticker["last"] if ticker else t["entry_price"]
        pnl_p  = (cur - t["entry_price"]) / t["entry_price"] * 100
        rows.append({
            "Symbol":     sym,
            "Entry $":    f'{t["entry_price"]:.8f}',
            "Current $":  f'{cur:.8f}',
            "TP $":       f'{t["tp"]:.8f}',
            "Trail SL $": f'{t["trail_sl"]:.8f}',
            "PnL %":      f'{pnl_p:+.3f}%',
            "Status":     "🟢 OPEN",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div style="text-align:center;padding:20px;color:#4a6280;font-size:13px;">'
        '⊘ No open positions — bot scanning for entries...</div>',
        unsafe_allow_html=True,
    )

# ── SCANNING LIST ──
st.markdown('<div class="section-header">🔍 AI SCANNING LIST</div>', unsafe_allow_html=True)
if st.session_state.scan_list:
    scan_rows = []
    for s in st.session_state.scan_list:
        scan_rows.append({
            "Symbol":    s["symbol"],
            "Price $":   f'{s["price"]:.8f}',
            "MFI":       s["mfi"],
            "Vol Ratio": f'×{s["vol_ratio"]}',
            "Momentum":  f'{s["momentum"]:+.3f}%',
            "AI Prob":   f'{s["ai_prob"]}%',
            "Signal":    s["signal"],
        })
    st.dataframe(pd.DataFrame(scan_rows), use_container_width=True, hide_index=True)
    st.markdown(
        f'<div style="font-size:10px;color:#4a6280;text-align:right;margin-top:4px;">'
        f'Last scan: {st.session_state.last_update}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="text-align:center;padding:20px;color:#4a6280;font-size:13px;">'
        '⊘ Start bot to begin scanning...</div>',
        unsafe_allow_html=True,
    )

# ── TRADE HISTORY ──
st.markdown('<div class="section-header">📋 TRADE HISTORY</div>', unsafe_allow_html=True)
if st.session_state.trades:
    hist_rows = []
    for t in reversed(st.session_state.trades[-50:]):
        pnl = t.get("pnl_pct", 0) or 0
        hist_rows.append({
            "Symbol":     t["symbol"],
            "Entry Time": t["entry_time"],
            "Entry $":    f'{t["entry_price"]:.8f}',
            "Exit $":     f'{t["exit_price"]:.8f}' if t["exit_price"] else "—",
            "PnL %":      f'{pnl:+.3f}%',
            "Status":     "🟢 WIN" if t["status"] == "WIN" else "🔴 LOSS",
        })
    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div style="text-align:center;padding:20px;color:#4a6280;font-size:13px;">'
        '⊘ No completed trades yet.</div>',
        unsafe_allow_html=True,
    )

# ── STATUS LOGS ──
st.markdown('<div class="section-header">📡 STATUS LOGS</div>', unsafe_allow_html=True)
if st.session_state.logs:
    log_html = '<div class="log-box">'
    for entry in st.session_state.logs:
        log_html += f'<div class="log-{entry["kind"]}">[{entry["ts"]}] {entry["msg"]}</div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="log-box" style="color:#4a6280;text-align:center;">'
        'Waiting for activity...</div>',
        unsafe_allow_html=True,
    )

# ── AI STATUS ──
st.markdown('<div class="section-header">🤖 AI MODEL STATUS</div>', unsafe_allow_html=True)
a1, a2, a3 = st.columns(3)
trained = st.session_state.ai_trained
with a1:
    st.markdown(
        f'<div class="metric-card {"green" if trained else "yellow"}">'
        f'<div class="metric-label">Model Status</div>'
        f'<div class="metric-value" style="font-size:16px;">{"✓ TRAINED" if trained else "○ LEARNING"}</div>'
        f'<div class="metric-sub">GradientBoosting</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with a2:
    n = len(st.session_state.ai_history)
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Training Samples</div>'
        f'<div class="metric-value">{n}</div>'
        f'<div class="metric-sub">Need 20 to train</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with a3:
    st.markdown(
        '<div class="metric-card">'
        '<div class="metric-label">Entry Threshold</div>'
        '<div class="metric-value">60%</div>'
        '<div class="metric-sub">AI confidence min</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── FOOTER ──
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:10px;color:#2a4060;letter-spacing:2px;">'
    '⚡ MEXC SCALPING BOT — PAPER TRADING ONLY — NOT FINANCIAL ADVICE ⚡'
    '</div>',
    unsafe_allow_html=True,
)

# ── AUTO REFRESH ──
if st.session_state.running:
    time.sleep(10)
    st.rerun()
📄 requirements.txt
streamlit>=1.32.0
ccxt>=4.3.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
📄 railway.toml
toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
طريقة التشغيل المحلي
bash
pip install streamlit ccxt pandas numpy scikit-learn
streamlit run app.py
الكود الآن أمامك كاملاً، انسخه مباشرة ✅

You said: هل يمكنك اخباري بطريقة رفعه وتشغيله لاني للاسف لا افقه شيى في البرمجه
هل يمكنك اخباري بطريقة رفعه وتشغيله لاني للاسف لا افقه شيى في البرمجه

Show more
27 أبريل
Claude responded: بالتأكيد!
بالتأكيد! سأشرح لك بأبسط طريقة ممكنة خطوة بخطوة 😊

أسهل طريقة — Streamlit Cloud (مجاني 100%)
الخطوة 1️⃣ — إنشاء حساب GitHub
اذهب لـ github.com
اضغط Sign up
أنشئ حساب مجاني بأي إيميل
الخطوة 2️⃣ — رفع الملفات على GitHub
بعد تسجيل الدخول اضغط "New repository" (الزر الأخضر)
سمّه مثلاً mexc-bot
اضغط "Create repository"
اضغط "uploading an existing file"
أنشئ 3 ملفات بهذه الأسماء بالضبط:
app.py ← الصق فيه كود البوت
requirements.txt ← الصق فيه المتطلبات
railway.toml ← الصق فيه إعدادات النشر
اضغط "Commit changes"
الخطوة 3️⃣ — رفعه على Streamlit Cloud
اذهب لـ share.streamlit.io
سجل دخول بنفس حساب GitHub
اضغط "New app"
اختر الـ repository اللي أنشأته mexc-bot
في خانة Main file path اكتب: app.py
اضغط "Deploy" ✅
بعد دقيقتين سيعطيك رابط تفتح منه البوت من أي جهاز أو موبايل 🎉

⚠️ ملاحظة مهمة
البوت حالياً يعمل بـ أموال وهمية فقط — لا يوجد أي خطر مالي. جرّبه وراقب كيف يعمل قبل أي خطوة أخرى.

هل تريد مني أشرح أي خطوة بتفصيل أكثر؟ 😊

You said: اريد معرفة شئ واحد اخر وهو هل سيعمل البوت بهذه الطريقه حتى وان اغلقة الانترنت في جهازي او اغلقة الكروم
اريد معرفة شئ واحد اخر وهو هل سيعمل البوت بهذه الطريقه حتى وان اغلقة الانترنت في جهازي او اغلقة الكروم

Show more
27 أبريل
Claude responded: نعم!
نعم! سيعمل بدونك تماماً ✅
لأنك رفعته على سيرفر Streamlit Cloud وليس على جهازك، يعني:

🟢 سيستمر في العمل حتى لو:
أغلقت الإنترنت على جهازك
أغلقت المتصفح
أغلقت الجهاز كلياً
نمت أو سافرت 😄
⚠️ لكن فيه تحذير مهم واحد
Streamlit Cloud المجاني يوقف التطبيق تلقائياً إذا لم يفتحه أحد لمدة 3-7 أيام

الحل بطريقتين:
الطريقة 1 — المجانية: موقع مثل uptimerobot.com يدخل على رابط البوت كل 5 دقائق تلقائياً ويبقيه مستيقظاً، وهو مجاني كامل

الطريقة 2 — Railway (مدفوع $5/شهر): لا يوقف أبداً، يعمل 24/7 بدون انقطاع

خلاصة سريعة:
الموقف	هل البوت يعمل؟
أغلقت المتصفح	✅ نعم
أغلقت الإنترنت	✅ نعم
أغلقت الجهاز	✅ نعم
لم يفتحه أحد أسبوع	⚠️ يتوقف تلقائياً
مع UptimeRobot	✅ يعمل للأبد مجاناً
You said: شكرا جزيلا
شكرا جزيلا
اخر سؤال طبعا انت لاتتحمل اي مسؤليه تجاه الاجابه
ولاكن لنفترض ان البوت عمل بشكل متواصل لمدة 5 ايام ثم قمت بتداول حقيقي للاموال واعطيته 5 دولارات مامدى نسبة نجاحه وخسارته بنسبه لك حاليا وخلال كم افتراضيا سيستطيع مضاعفة المال 
اريد هذه المعلومات فقط من باب المعرفه لاغير لاكي افكر في طريقه لتحسينه مستقبلا

Show more
27 أبريل
Claude responded: تقييم صادق للبوت 📊
تقييم صادق للبوت 📊
أولاً — الواقع الحالي بصدق
النسبة المتوقعة تقريباً:

المقياس	التوقع الواقعي
Win Rate	40% - 55%
الصفقات الخاسرة	45% - 60%
متوسط ربح الصفقة	1.5%
متوسط خسارة الصفقة	1.2%
ثانياً — مشكلة الـ $5 تحديداً
هذا المبلغ فيه عقبة عملية قبل حتى الحديث عن الاستراتيجية:

رسوم MEXC على كل صفقة تتراوح بين 0.1% و 0.2%
يعني كل صفقة ذهاباً وإياباً تأكل تقريباً 0.3% - 0.4% من المبلغ
على $5 هذا يعني كل صفقة تكلفك $0.015 - $0.02 قبل أي ربح أو خسارة
البوت يفتح صفقات كثيرة = الرسوم تتراكم بسرعة
ثالثاً — السبب الأكبر للقلق
العملات بسعر $0.001 أو أقل لها طبيعة خاصة جداً:

السيولة منخفضة = صعب الخروج بالسعر المطلوب بالضبط
التلاعب سهل = مجموعات الـ Pump & Dump تستهدفها تحديداً
الفارق بين سعر الشراء والبيع (Spread) قد يأكل الربح كاملاً
رابعاً — تقدير مضاعفة الـ $5
بصراحة تامة، في السيناريو التفاؤلي جداً:

الأسبوع 1:  $5.00 → $5.80  (+16%)
الشهر 1:    $5.80 → $8.00  (إذا سارت الأمور جيداً)
في السيناريو الواقعي الأكثر احتمالاً:

بعد أسبوعين:  $5.00 → $3.50 → $4.20  (تذبذب)
النتيجة الغالبة: خسارة 20%-40% من المبلغ
خامساً — ما يحتاج تحسيناً فعلاً قبل التداول الحقيقي
هذه النقاط هي التي ستصنع الفرق:

1. Backtesting أولاً اختبر الاستراتيجية على بيانات تاريخية قبل أي دولار حقيقي

2. حساب الرسوم داخل البوت البوت الحالي لا يحسب رسوم MEXC في قرار الدخول

3. رفع حد الدخول بدلاً من 60% ثقة AI، جرّب 75% لتقليل الصفقات الخاسرة

4. تصفية العملات إضافة فلتر يستبعد العملات ذات السيولة المنخفضة جداً

الخلاصة الصادقة
إذا كنت تفكر في تحسينه مستقبلاً، فالأولوية هي Backtesting على 30 يوم من البيانات الحقيقية قبل أي تداول حقيقي — هذا وحده سيعطيك صورة أوضح بكثير من أي تقدير نظري 🎯

You said: هل يمكننا تطوير البوت ليصبح اقوى وافضل من ناحية ادخال الارباح وايضا اضافة العموله في حسبان البوت كما ذكرت لي سابقا ان يكون البوت باللغه العربيه يستحسا هذا
هل يمكننا تطوير البوت ليصبح اقوى وافضل من ناحية ادخال الارباح وايضا اضافة العموله في حسبان البوت كما ذكرت لي سابقا ان يكون البوت باللغه العربيه يستحسا هذا 
في الواجهه اريد اضفة شيئ مهم وهو في البوت الحالي لا أستطيع ان ارى ماذا يفعل البوت انتظر ان يدخل في صفقه او يحدث اي شيئ الا انه لمدة ساعه لم يتغير شيئ
اريد مثلا اسم العمله التي يستكشفها ثم يخبرني بانها لاتصلح مثلا اظهار اسم العمله التاليه وهاكذا ايضا يمكنك اضافة اي ميزات تخدم في صالح البوت دون ان يضر في المهمه الاساسيه

Show more
12:47 ص
Claude responded: سأبني النسخة المحسّنة كاملة الآن!
سأبني النسخة المحسّنة كاملة الآن!


Check skills available
Check skills available
الآن سأكتب الكود المحسّن الكامل:

python
"""
╔══════════════════════════════════════════════════════════════╗
║        بوت المضاربة السريعة — MEXC Scalping Bot v2.0        ║
║     استراتيجية: انفجار السيولة + MFI + ذكاء اصطناعي        ║
║              وضع: تداول وهمي — Paper Trading                ║
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
#  إعداد الصفحة
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="بوت المضاربة السريعة",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  التصميم البصري الداكن — الثيم العربي
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Cairo:wght@400;600;700;900&display=swap');

:root {
    --bg-deep:   #060b10;
    --bg-card:   #0b1520;
    --bg-panel:  #0f1e2e;
    --border:    #1a3050;
    --accent:    #00c8ff;
    --green:     #00ff9d;
    --red:       #ff3860;
    --yellow:    #ffd600;
    --orange:    #ff8c00;
    --text:      #c8ddf0;
    --text-dim:  #4a6a8a;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    color: var(--text) !important;
    font-family: 'Cairo', sans-serif !important;
    direction: rtl;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-left: 1px solid var(--border) !important;
}
.block-container { padding: 1rem 1.5rem !important; }

/* بطاقات المقاييس */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
    margin-bottom: 8px;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 100%; height: 3px;
    background: var(--accent);
}
.metric-card.green::after  { background: var(--green); }
.metric-card.red::after    { background: var(--red); }
.metric-card.yellow::after { background: var(--yellow); }
.metric-card.orange::after { background: var(--orange); }
.metric-label {
    font-size: 11px; letter-spacing: 1px;
    color: var(--text-dim); margin-bottom: 6px;
    font-family: 'Cairo', sans-serif; font-weight: 600;
}
.metric-value {
    font-size: 24px; font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.metric-sub { font-size: 11px; color: var(--text-dim); margin-top: 5px; }

/* عناوين الأقسام */
.section-header {
    font-family: 'Cairo', sans-serif;
    font-size: 14px; font-weight: 700;
    letter-spacing: 1px;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px; margin: 22px 0 12px;
    display: flex; align-items: center; gap: 8px;
}

/* بطاقة الفحص المباشر */
.scan-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
    transition: border-color .3s;
}
.scan-card.active { border-color: var(--accent); }
.scan-card.pass   { border-color: var(--green); }
.scan-card.fail   { border-color: #1a3050; opacity: .7; }
.scan-symbol {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px; font-weight: 700; color: var(--text);
}
.scan-reason {
    font-size: 12px; color: var(--text-dim);
    font-family: 'Cairo', sans-serif;
}
.scan-badge {
    font-size: 11px; padding: 3px 10px;
    border-radius: 20px; font-weight: 700;
    font-family: 'Cairo', sans-serif;
}
.badge-scanning { background:#0a2040; color:var(--accent); border:1px solid var(--accent); }
.badge-pass     { background:#002a1a; color:var(--green);  border:1px solid var(--green); }
.badge-fail     { background:#1a0010; color:var(--red);    border:1px solid var(--red); }
.badge-skip     { background:#1a1a00; color:var(--yellow); border:1px solid var(--yellow); }

/* صندوق السجلات */
.log-box {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 12px;
    max-height: 220px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    direction: ltr;
}
.log-buy   { color: var(--green); }
.log-sell  { color: var(--red); }
.log-info  { color: var(--accent); }
.log-warn  { color: var(--yellow); }
.log-scan  { color: #3a6a8a; }
.log-pass  { color: #00cc7a; }
.log-fail  { color: #aa2244; }

/* أزرار التحكم */
.stButton > button {
    font-family: 'Cairo', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    background: transparent !important;
    transition: all .2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg-deep) !important;
}
.start-btn > button { border-color: var(--green) !important; color: var(--green) !important; }
.start-btn > button:hover { background: var(--green) !important; color: var(--bg-deep) !important; }
.stop-btn  > button { border-color: var(--red) !important;   color: var(--red) !important; }
.stop-btn  > button:hover { background: var(--red) !important;   color: var(--bg-deep) !important; }

/* الحقول */
.stNumberInput input, .stSelectbox select {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px !important;
}

/* نبضة الحالة */
.pulse {
    display:inline-block; width:9px; height:9px;
    border-radius:50%; background:var(--green);
    animation: blink 1.2s infinite; margin-left:6px;
}
@keyframes blink {
    0%,100%{opacity:1;transform:scale(1)}
    50%{opacity:.3;transform:scale(.6)}
}
.pulse.off { background:var(--red); animation:none; }

/* شريط قوة الإشارة */
.signal-bar-wrap { display:flex; align-items:center; gap:6px; }
.signal-bar {
    height:5px; border-radius:3px;
    background:linear-gradient(90deg,var(--accent),var(--green));
    transition: width .4s;
}

/* الجداول */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* شريط العنوان الجانبي */
.sidebar-logo {
    text-align:center; padding:10px 0 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}
.sidebar-logo .title {
    font-family:'Cairo',sans-serif;
    font-size:20px; font-weight:900;
    color:var(--accent); letter-spacing:1px;
}
.sidebar-logo .sub {
    font-size:11px; color:var(--text-dim); margin-top:2px;
}

/* تنبيه مؤقت */
.flash-alert {
    background: #002a1a;
    border: 1px solid var(--green);
    border-radius:8px; padding:10px 16px;
    font-family:'Cairo',sans-serif;
    font-size:13px; color:var(--green);
    margin-bottom:8px;
    animation: fadeout 5s forwards;
}
@keyframes fadeout { 0%{opacity:1} 80%{opacity:1} 100%{opacity:0} }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  تهيئة حالة الجلسة
# ════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "running":       False,
        "budget":        100.0,
        "balance":       100.0,
        "trades":        [],
        "scan_list":     [],
        "open_trades":   {},
        "logs":          deque(maxlen=80),
        "ai_model":      None,
        "ai_scaler":     None,
        "ai_trained":    False,
        "ai_history":    [],
        "stop_event":    threading.Event(),
        "stats": {"win":0,"loss":0,"pnl_today":0.0,"pnl_hour":0.0,
                  "total_fees":0.0,"best_trade":0.0,"worst_trade":0.0},
        "last_update":   None,
        "current_scan":  "",
        "scan_results":  deque(maxlen=30),
        "alerts":        deque(maxlen=5),
        "commision_rate": 0.002,   # 0.2% لكل اتجاه
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ════════════════════════════════════════════════════════════
#  الثوابت والإعدادات
# ════════════════════════════════════════════════════════════
MAX_OPEN        = 2
TAKE_PROFIT_PCT = 0.018    # 1.8% هدف ربح
INITIAL_SL_PCT  = 0.012    # 1.2% وقف خسارة أولي
TRAILING_DIST   = 0.008    # 0.8% مسافة التتبع
MFI_BUY_THRESH  = 58       # عتبة MFI للدخول
VOL_MULTIPLIER  = 2.2      # مضاعف حجم التداول
SCAN_INTERVAL   = 6        # ثواني بين كل دورة فحص
MAX_PRICE       = 0.001    # أقصى سعر للعملة
MIN_AI_PROB     = 0.62     # أدنى ثقة للذكاء الاصطناعي
COMMISSION      = 0.002    # 0.2% عمولة لكل اتجاه

WATCHLIST = [
    "PEPE/USDT","SHIB/USDT","FLOKI/USDT","BONK/USDT","WIF/USDT",
    "LUNC/USDT","XEC/USDT","HOT/USDT","BTT/USDT","WIN/USDT",
    "VOLT/USDT","SAMO/USDT","BABYDOGE/USDT","TURBO/USDT","RATS/USDT",
    "NEIRO/USDT","MOG/USDT","MEME/USDT","MYRO/USDT","SLERF/USDT",
]

# أسباب رفض الدخول
REJECT_REASONS = {
    "price":    "السعر أعلى من $0.001",
    "mfi":      "MFI منخفض — لا تدفق للأموال",
    "volume":   "حجم التداول ضعيف",
    "ai":       "ثقة الذكاء الاصطناعي منخفضة",
    "open":     "وصلنا للحد الأقصى من الصفقات",
    "balance":  "الرصيد غير كافٍ",
    "data":     "بيانات غير كافية",
    "fee":      "الربح المتوقع لا يغطي العمولة",
}

# ════════════════════════════════════════════════════════════
#  الاتصال بالبورصة
# ════════════════════════════════════════════════════════════
@st.cache_resource
def get_exchange():
    return ccxt.mexc({"enableRateLimit": True})

exchange = get_exchange()

# ════════════════════════════════════════════════════════════
#  المؤشرات الفنية
# ════════════════════════════════════════════════════════════
def compute_mfi(highs, lows, closes, volumes, period=14):
    tps = (np.array(highs) + np.array(lows) + np.array(closes)) / 3
    raw_mf = tps * np.array(volumes)
    pos = np.zeros(len(tps))
    neg = np.zeros(len(tps))
    for i in range(1, len(tps)):
        (pos if tps[i] > tps[i-1] else neg)[i] = raw_mf[i]
    vals = []
    for i in range(period, len(tps)):
        p = pos[i-period:i].sum()
        n = neg[i-period:i].sum()
        vals.append(100.0 if n == 0 else 100 - 100/(1 + p/n))
    return vals[-1] if vals else 50.0

def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g  = np.mean(gains[-period:])
    avg_l  = np.mean(losses[-period:])
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100 - 100 / (1 + rs)

def compute_momentum(closes, period=5):
    if len(closes) < period + 1:
        return 0.0
    return (closes[-1] - closes[-period-1]) / closes[-period-1] * 100

def compute_atr(highs, lows, closes, period=14):
    trs = [
        max(highs[i]-lows[i],
            abs(highs[i]-closes[i-1]),
            abs(lows[i]-closes[i-1]))
        for i in range(1, len(closes))
    ]
    return np.mean(trs[-period:]) if trs else closes[-1] * 0.01

def compute_volume_ratio(volumes, lookback=20):
    if len(volumes) < lookback + 1:
        return 1.0
    avg = np.mean(volumes[-lookback-1:-1])
    return 1.0 if avg == 0 else volumes[-1] / avg

def compute_ema(closes, period=20):
    if len(closes) < period:
        return closes[-1]
    k = 2 / (period + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = c * k + ema * (1 - k)
    return ema

# ════════════════════════════════════════════════════════════
#  الذكاء الاصطناعي
# ════════════════════════════════════════════════════════════
def get_features(mfi, vol_ratio, momentum, atr_pct, rsi, ema_dist):
    return [mfi, vol_ratio, momentum, atr_pct, rsi, ema_dist]

def ai_predict(features):
    m  = st.session_state.ai_model
    sc = st.session_state.ai_scaler
    if m is None or not st.session_state.ai_trained:
        mfi, vol_ratio, momentum, atr_pct, rsi, ema_dist = features
        score = 0.0
        if mfi > MFI_BUY_THRESH:       score += 0.30
        if vol_ratio > VOL_MULTIPLIER:  score += 0.30
        if momentum > 0.3:              score += 0.15
        if rsi > 50 and rsi < 75:      score += 0.15
        if ema_dist > 0:               score += 0.10
        return min(score, 0.99)
    try:
        X = sc.transform([features])
        return float(m.predict_proba(X)[0][1])
    except Exception:
        return 0.5

def ai_retrain():
    hist = st.session_state.ai_history
    if len(hist) < 20:
        return
    X = [h[0] for h in hist]
    y = [h[1] for h in hist]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.08, subsample=0.8,
        min_samples_leaf=3, random_state=42,
    )
    try:
        model.fit(Xs, y)
        st.session_state.ai_model   = model
        st.session_state.ai_scaler  = sc
        st.session_state.ai_trained = True
        add_log("✓ تم تدريب الذكاء الاصطناعي على بيانات جديدة", "info")
    except Exception:
        pass

def ai_record(features, won: bool):
    st.session_state.ai_history.append((features, int(won)))
    if len(st.session_state.ai_history) % 10 == 0:
        ai_retrain()

# ════════════════════════════════════════════════════════════
#  نظام السجلات
# ════════════════════════════════════════════════════════════
def add_log(msg: str, kind: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.appendleft({"ts": ts, "msg": msg, "kind": kind})

def add_scan_result(symbol: str, status: str, reason: str, ai_prob: float = 0):
    st.session_state.scan_results.appendleft({
        "symbol":  symbol,
        "status":  status,   # scanning / pass / fail / skip
        "reason":  reason,
        "ai_prob": ai_prob,
        "time":    datetime.now().strftime("%H:%M:%S"),
    })

def add_alert(msg: str):
    st.session_state.alerts.appendleft({
        "msg":  msg,
        "time": datetime.now().strftime("%H:%M:%S"),
    })

# ════════════════════════════════════════════════════════════
#  جلب البيانات
# ════════════════════════════════════════════════════════════
def fetch_ohlcv(symbol, timeframe="1m", limit=60):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not data or len(data) < 20:
            return None
        return pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    except Exception:
        return None

def fetch_ticker(symbol):
    try:
        return exchange.fetch_ticker(symbol)
    except Exception:
        return None

# ════════════════════════════════════════════════════════════
#  منطق التداول — مع حساب العمولة
# ════════════════════════════════════════════════════════════
def calc_min_profit_pct():
    """الحد الأدنى للربح بعد خصم العمولة ذهاباً وإياباً"""
    return COMMISSION * 2 + 0.003   # عمولتان + هامش أمان 0.3%

def open_trade(symbol, price, atr, features):
    slots = MAX_OPEN - len(st.session_state.open_trades)
    if slots <= 0:
        return
    alloc     = st.session_state.balance / slots
    fee_entry = alloc * COMMISSION
    qty       = (alloc - fee_entry) / price
    sl        = price * (1 - INITIAL_SL_PCT)
    tp        = price * (1 + TAKE_PROFIT_PCT)

    trade = {
        "symbol":      symbol,
        "entry_price": price,
        "qty":         qty,
        "alloc":       alloc,
        "fee_entry":   fee_entry,
        "sl":          sl,
        "tp":          tp,
        "trail_high":  price,
        "trail_sl":    sl,
        "entry_time":  datetime.now().isoformat(timespec="seconds"),
        "exit_price":  None,
        "exit_time":   None,
        "pnl_pct":     None,
        "pnl_net":     None,
        "status":      "OPEN",
        "features":    features,
        "atr":         atr,
    }
    st.session_state.open_trades[symbol] = trade
    st.session_state.balance -= alloc
    st.session_state.stats["total_fees"] = round(
        st.session_state.stats["total_fees"] + fee_entry, 6)

    add_log(f"▲ شراء {symbol} @ ${price:.8f} | وقف={sl:.8f} | هدف={tp:.8f}", "buy")
    add_alert(f"✅ تم الشراء: {symbol} @ ${price:.8f}")
    add_scan_result(symbol, "pass", f"دخلنا الصفقة — AI: {ai_predict(features)*100:.0f}%", ai_predict(features))

def close_trade(symbol, price, reason=""):
    trade = st.session_state.open_trades.pop(symbol, None)
    if not trade:
        return

    fee_exit  = trade["qty"] * price * COMMISSION
    gross_pnl = (price - trade["entry_price"]) * trade["qty"]
    net_pnl   = gross_pnl - trade["fee_entry"] - fee_exit
    pnl_pct   = net_pnl / trade["alloc"] * 100

    st.session_state.balance += trade["qty"] * price - fee_exit
    st.session_state.stats["total_fees"] = round(
        st.session_state.stats["total_fees"] + fee_exit, 6)

    trade.update({
        "exit_price": price,
        "exit_time":  datetime.now().isoformat(timespec="seconds"),
        "pnl_pct":    round(pnl_pct, 3),
        "pnl_net":    round(net_pnl, 6),
        "status":     "WIN" if pnl_pct > 0 else "LOSS",
    })
    st.session_state.trades.append(trade)

    # تحديث الإحصائيات
    s = st.session_state.stats
    s["pnl_today"] = round(s["pnl_today"] + net_pnl, 6)
    s["pnl_hour"]  = round(s["pnl_hour"]  + net_pnl, 6)
    if pnl_pct > 0:
        s["win"] += 1
        s["best_trade"] = max(s["best_trade"], pnl_pct)
        add_log(f"▼ بيع {symbol} @ ${price:.8f} | ربح صافٍ: {pnl_pct:+.2f}% | [{reason}]", "buy")
        add_alert(f"💰 ربح: {symbol} {pnl_pct:+.2f}% بعد العمولة")
    else:
        s["loss"] += 1
        s["worst_trade"] = min(s["worst_trade"], pnl_pct)
        add_log(f"▼ بيع {symbol} @ ${price:.8f} | خسارة: {pnl_pct:+.2f}% | [{reason}]", "sell")
        add_alert(f"⛔ خسارة: {symbol} {pnl_pct:+.2f}% — وقف الخسارة")

    ai_record(trade["features"], pnl_pct > 0)

def manage_open_trades():
    for sym in list(st.session_state.open_trades.keys()):
        t      = st.session_state.open_trades[sym]
        ticker = fetch_ticker(sym)
        if not ticker:
            continue
        price = ticker["last"]
        # تحديث وقف الخسارة المتحرك
        if price > t["trail_high"]:
            t["trail_high"] = price
            t["trail_sl"]   = price * (1 - TRAILING_DIST)
        # شروط الخروج
        if price >= t["tp"]:
            close_trade(sym, price, "هدف الربح ✓")
        elif price <= t["trail_sl"]:
            close_trade(sym, price, "وقف متحرك")
        elif price <= t["sl"]:
            close_trade(sym, price, "وقف الخسارة")

# ════════════════════════════════════════════════════════════
#  محرك الفحص الذكي
# ════════════════════════════════════════════════════════════
def scan_symbols():
    results = []
    min_profit = calc_min_profit_pct()

    for sym in WATCHLIST:
        # عرض اسم العملة الحالية في الفحص
        st.session_state.current_scan = sym
        add_log(f"⟳ فحص {sym}...", "scan")

        try:
            # ── فحص الحد الأقصى للصفقات ──
            if len(st.session_state.open_trades) >= MAX_OPEN and sym not in st.session_state.open_trades:
                add_scan_result(sym, "skip", REJECT_REASONS["open"])
                continue

            ticker = fetch_ticker(sym)
            if not ticker:
                add_scan_result(sym, "fail", REJECT_REASONS["data"])
                continue

            price = ticker["last"]

            # ── فحص السعر ──
            if price is None or price > MAX_PRICE:
                add_scan_result(sym, "fail", f"{REJECT_REASONS['price']} (${price:.6f})")
                add_log(f"✗ {sym} — {REJECT_REASONS['price']}", "fail")
                continue

            df = fetch_ohlcv(sym)
            if df is None:
                add_scan_result(sym, "fail", REJECT_REASONS["data"])
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
            ema_dist  = (closes[-1] - ema20) / ema20 * 100
            features  = get_features(mfi, vol_ratio, momentum, atr_pct, rsi, ema_dist)
            ai_prob   = ai_predict(features)

            # ── فحص MFI ──
            if mfi <= MFI_BUY_THRESH:
                reason = f"{REJECT_REASONS['mfi']} (MFI={mfi:.1f})"
                add_scan_result(sym, "fail", reason, ai_prob)
                add_log(f"✗ {sym} — {reason}", "fail")
                results.append(build_result(sym, price, mfi, vol_ratio, momentum, ai_prob, "○ ضعيف", features, atr))
                continue

            # ── فحص حجم التداول ──
            if vol_ratio <= VOL_MULTIPLIER:
                reason = f"{REJECT_REASONS['volume']} (×{vol_ratio:.1f})"
                add_scan_result(sym, "fail", reason, ai_prob)
                add_log(f"✗ {sym} — {reason}", "fail")
                results.append(build_result(sym, price, mfi, vol_ratio, momentum, ai_prob, "○ ضعيف", features, atr))
                continue

            # ── فحص العمولة ──
            expected_profit_pct = TAKE_PROFIT_PCT * 100
            if expected_profit_pct < min_profit * 100:
                reason = f"{REJECT_REASONS['fee']} (متوقع={expected_profit_pct:.2f}% < لازم={min_profit*100:.2f}%)"
                add_scan_result(sym, "fail", reason, ai_prob)
                add_log(f"✗ {sym} — {reason}", "fail")
                continue

            # ── فحص الذكاء الاصطناعي ──
            if ai_prob < MIN_AI_PROB:
                reason = f"{REJECT_REASONS['ai']} ({ai_prob*100:.0f}%)"
                add_scan_result(sym, "fail", reason, ai_prob)
                add_log(f"✗ {sym} — {reason}", "fail")
                results.append(build_result(sym, price, mfi, vol_ratio, momentum, ai_prob, "◎ متابعة", features, atr))
                continue

            # ── اجتاز كل الفلاتر ✓ ──
            signal = "⚡ قوي"
            add_scan_result(sym, "pass", f"اجتاز جميع الفلاتر — AI={ai_prob*100:.0f}%", ai_prob)
            add_log(f"✓ {sym} — إشارة قوية! MFI={mfi:.0f} Vol=×{vol_ratio:.1f} AI={ai_prob*100:.0f}%", "pass")
            results.append(build_result(sym, price, mfi, vol_ratio, momentum, ai_prob, signal, features, atr))

            # ── قرار الدخول ──
            if (sym not in st.session_state.open_trades
                    and len(st.session_state.open_trades) < MAX_OPEN
                    and st.session_state.balance > 1.0):
                open_trade(sym, price, atr, features)

        except Exception as e:
            add_scan_result(sym, "fail", f"خطأ: {str(e)[:40]}")
            add_log(f"⚠ خطأ في {sym}: {str(e)[:50]}", "warn")
            continue

    results.sort(key=lambda x: x["ai_prob"], reverse=True)
    st.session_state.scan_list   = results
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
    st.session_state.current_scan = ""

def build_result(sym, price, mfi, vol_ratio, momentum, ai_prob, signal, features, atr):
    return {
        "symbol":    sym,
        "price":     price,
        "mfi":       round(mfi, 1),
        "vol_ratio": round(vol_ratio, 2),
        "momentum":  round(momentum, 3),
        "ai_prob":   round(ai_prob * 100, 1),
        "signal":    signal,
        "features":  features,
        "atr":       atr,
    }

# ════════════════════════════════════════════════════════════
#  خيط البوت الرئيسي
# ════════════════════════════════════════════════════════════
def bot_loop(stop_event: threading.Event):
    add_log("⚡ بدأ البوت — وضع التداول الوهمي", "info")
    while not stop_event.is_set():
        try:
            manage_open_trades()
            scan_symbols()
        except Exception as e:
            add_log(f"⚠ خطأ عام: {e}", "warn")
        for _ in range(SCAN_INTERVAL * 10):
            if stop_event.is_set():
                break
            time.sleep(0.1)
    add_log("■ توقف البوت.", "warn")

def start_bot():
    if st.session_state.running:
        return
    st.session_state.stop_event.clear()
    st.session_state.balance     = st.session_state.budget
    st.session_state.trades      = []
    st.session_state.open_trades = {}
    st.session_state.scan_results = deque(maxlen=30)
    st.session_state.stats = {
        "win":0,"loss":0,"pnl_today":0.0,"pnl_hour":0.0,
        "total_fees":0.0,"best_trade":0.0,"worst_trade":0.0
    }
    st.session_state.logs   = deque(maxlen=80)
    st.session_state.alerts = deque(maxlen=5)
    t = threading.Thread(
        target=bot_loop,
        args=(st.session_state.stop_event,),
        daemon=True,
    )
    t.start()
    st.session_state.running = True

def stop_bot():
    if not st.session_state.running:
        return
    st.session_state.stop_event.set()
    st.session_state.running = False

# ════════════════════════════════════════════════════════════
#  الشريط الجانبي
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="title">⚡ بوت المضاربة</div>
        <div class="sub">MEXC Scalping Bot v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    running  = st.session_state.running
    dot_cls  = "pulse" if running else "pulse off"
    stat_ar  = "يعمل الآن" if running else "متوقف"
    st.markdown(
        f'<div style="text-align:center;margin-bottom:14px;font-family:Cairo,sans-serif;">'
        f'<span style="font-size:14px;font-weight:700;">{stat_ar}</span>'
        f'<span class="{dot_cls}"></span></div>',
        unsafe_allow_html=True,
    )

    budget = st.number_input(
        "💰 الميزانية (USDT)",
        min_value=10.0, max_value=10000.0,
        value=float(st.session_state.budget), step=10.0,
    )
    st.session_state.budget = budget

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        if st.button("▶ تشغيل", use_container_width=True):
            start_bot(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        if st.button("■ إيقاف", use_container_width=True):
            stop_bot(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # الرصيد الحالي
    bal     = st.session_state.balance
    ini     = st.session_state.budget
    bal_pct = (bal - ini) / ini * 100 if ini > 0 else 0
    col     = "#00ff9d" if bal >= ini else "#ff3860"
    st.markdown(
        f'<div class="metric-label">الرصيد الحالي</div>'
        f'<div style="font-size:24px;font-weight:900;color:{col};font-family:JetBrains Mono,monospace;">'
        f'${bal:.4f}</div>'
        f'<div style="font-size:12px;color:{col};margin-bottom:12px;">{bal_pct:+.2f}% من البداية</div>',
        unsafe_allow_html=True,
    )

    # إجمالي العمولات
    fees = st.session_state.stats["total_fees"]
    st.markdown(
        f'<div class="metric-label">إجمالي العمولات المدفوعة</div>'
        f'<div style="font-size:16px;font-weight:700;color:#ff8c00;font-family:JetBrains Mono,monospace;">'
        f'${fees:.6f}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px;color:#3a6a8a;line-height:2;">
    📊 انفجار السيولة + MFI<br>
    🤖 Gradient Boosting AI<br>
    🎯 هدف الربح: 1.8%<br>
    🛡 وقف الخسارة: 1.2%<br>
    🔄 الوقف المتحرك: 0.8%<br>
    💸 العمولة: 0.2% × 2<br>
    💰 أقصى سعر: $0.001<br>
    ⚡ أقصى صفقات: 2
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 تحديث الواجهة", use_container_width=True):
        st.rerun()

# ════════════════════════════════════════════════════════════
#  لوحة التحكم الرئيسية
# ════════════════════════════════════════════════════════════
st.markdown(
    '<h1 style="font-family:Cairo,sans-serif;font-size:26px;font-weight:900;'
    'color:#00c8ff;margin:0 0 2px;">🚀 مركز التحكم — بوت المضاربة السريعة</h1>'
    '<div style="font-size:12px;color:#3a6a8a;margin-bottom:18px;">'
    '⚡ وضع التداول الوهمي — أسعار MEXC المباشرة — مع حساب العمولة الكاملة</div>',
    unsafe_allow_html=True,
)

# ── تنبيهات لحظية ──
if st.session_state.alerts:
    for alert in list(st.session_state.alerts)[:2]:
        st.markdown(
            f'<div class="flash-alert">🔔 [{alert["time"]}] {alert["msg"]}</div>',
            unsafe_allow_html=True,
        )

# ── المقاييس العلوية ──
stats  = st.session_state.stats
total  = stats["win"] + stats["loss"]
wr     = stats["win"] / total * 100 if total > 0 else 0.0
pnl_d  = stats["pnl_today"]
pnl_h  = stats["pnl_hour"]

def metric_card(col, label, value, sub="", css=""):
    with col:
        st.markdown(
            f'<div class="metric-card {css}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

c1,c2,c3,c4,c5,c6 = st.columns(6)
metric_card(c1,"ربح اليوم",
    f'{"+" if pnl_d>=0 else ""}{pnl_d:.4f}$',
    "صافٍ بعد العمولة","green" if pnl_d>=0 else "red")
metric_card(c2,"ربح الساعة",
    f'{"+" if pnl_h>=0 else ""}{pnl_h:.4f}$',
    "متجدد","green" if pnl_h>=0 else "red")
metric_card(c3,"رابح / خاسر",
    f'{stats["win"]} / {stats["loss"]}',
    f'{total} إجمالي الصفقات',"yellow")
metric_card(c4,"نسبة النجاح",
    f'{wr:.1f}%',
    "AI يتعلم..." if not st.session_state.ai_trained else "AI فعّال ✓",
    "green" if wr>=50 else "red")
metric_card(c5,"الصفقات المفتوحة",
    f'{len(st.session_state.open_trades)}/{MAX_OPEN}',
    "الحد الأقصى صفقتان","orange")
metric_card(c6,"إجمالي العمولات",
    f'${stats["total_fees"]:.5f}',
    "مخصوم تلقائياً","red")

# ── شاشة الفحص المباشر ──
st.markdown('<div class="section-header">🔍 الفحص المباشر — ماذا يفعل البوت الآن؟</div>', unsafe_allow_html=True)

cur = st.session_state.current_scan
if cur:
    st.markdown(
        f'<div class="scan-card active">'
        f'<div><span class="scan-symbol">{cur}</span>'
        f'<div class="scan-reason">جارٍ التحليل...</div></div>'
        f'<span class="scan-badge badge-scanning">⟳ يفحص</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

scan_history = list(st.session_state.scan_results)[:15]
if scan_history:
    for r in scan_history:
        css  = {"pass":"pass","fail":"fail","skip":"fail","scanning":"active"}.get(r["status"],"fail")
        badge_cls = {"pass":"badge-pass","fail":"badge-fail","skip":"badge-skip","scanning":"badge-scanning"}.get(r["status"],"badge-fail")
        badge_txt = {"pass":"✓ اجتاز","fail":"✗ رُفض","skip":"⊘ تخطى","scanning":"⟳ يفحص"}.get(r["status"],"✗")
        bar = f'<div class="signal-bar-wrap"><div class="signal-bar" style="width:{int(r["ai_prob"]*60)}px"></div><span style="font-size:10px;color:#4a6a8a;">{r["ai_prob"]*100:.0f}%</span></div>' if r["ai_prob"] > 0 else ""
        st.markdown(
            f'<div class="scan-card {css}">'
            f'<div>'
            f'<span class="scan-symbol">{r["symbol"]}</span>'
            f'<div class="scan-reason">{r["reason"]}</div>'
            f'{bar}</div>'
            f'<div style="text-align:left;">'
            f'<span class="scan-badge {badge_cls}">{badge_txt}</span>'
            f'<div style="font-size:10px;color:#3a6a8a;margin-top:4px;">{r["time"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
elif not cur:
    st.markdown(
        '<div style="text-align:center;padding:24px;color:#3a6a8a;font-family:Cairo,sans-serif;">'
        'شغّل البوت لبدء الفحص التلقائي...</div>',
        unsafe_allow_html=True,
    )

# ── الصفقات المفتوحة ──
st.markdown('<div class="section-header">🔴 الصفقات المفتوحة حالياً</div>', unsafe_allow_html=True)
if st.session_state.open_trades:
    rows = []
    for sym, t in st.session_state.open_trades.items():
        ticker = fetch_ticker(sym)
        cur_p  = ticker["last"] if ticker else t["entry_price"]
        gross  = (cur_p - t["entry_price"]) * t["qty"]
        net    = gross - t["fee_entry"] - (t["qty"] * cur_p * COMMISSION)
        pnl_p  = net / t["alloc"] * 100
        rows.append({
            "العملة":         sym,
            "سعر الدخول":    f'${t["entry_price"]:.8f}',
            "السعر الحالي":  f'${cur_p:.8f}',
            "هدف الربح":     f'${t["tp"]:.8f}',
            "وقف متحرك":    f'${t["trail_sl"]:.8f}',
            "ربح/خسارة صافٍ": f'{pnl_p:+.3f}%',
            "الحالة":         "🟢 مفتوحة",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div style="text-align:center;padding:18px;color:#3a6a8a;font-family:Cairo,sans-serif;">'
        '⊘ لا توجد صفقات مفتوحة — البوت يبحث عن فرص...</div>',
        unsafe_allow_html=True,
    )

# ── قائمة الفحص الشاملة ──
st.markdown('<div class="section-header">📡 قائمة الفحص الشاملة — تصنيف الذكاء الاصطناعي</div>', unsafe_allow_html=True)
if st.session_state.scan_list:
    rows = []
    for s in st.session_state.scan_list:
        rows.append({
            "العملة":      s["symbol"],
            "السعر $":    f'{s["price"]:.8f}',
            "MFI":         s["mfi"],
            "حجم التداول": f'×{s["vol_ratio"]}',
            "الزخم":       f'{s["momentum"]:+.3f}%',
            "ثقة AI":      f'{s["ai_prob"]}%',
            "الإشارة":     s["signal"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(
        f'<div style="font-size:11px;color:#3a6a8a;margin-top:4px;font-family:Cairo,sans-serif;">'
        f'آخر تحديث: {st.session_state.last_update}</div>',
        unsafe_allow_html=True,
    )

# ── سجل الصفقات ──
st.markdown('<div class="section-header">📋 سجل الصفقات التفصيلي</div>', unsafe_allow_html=True)
if st.session_state.trades:
    rows = []
    for t in reversed(st.session_state.trades[-50:]):
        pnl = t.get("pnl_pct", 0) or 0
        net = t.get("pnl_net", 0) or 0
        rows.append({
            "العملة":       t["symbol"],
            "وقت الدخول":  t["entry_time"],
            "سعر الدخول":  f'${t["entry_price"]:.8f}',
            "سعر الخروج":  f'${t["exit_price"]:.8f}' if t["exit_price"] else "—",
            "الربح/الخسارة": f'{pnl:+.3f}%',
            "صافٍ (USDT)": f'{net:+.6f}',
            "الحالة":       "🟢 رابح" if t["status"]=="WIN" else "🔴 خاسر",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # إحصائية سريعة
    s = st.session_state.stats
    ia1, ia2, ia3 = st.columns(3)
    with ia1:
        st.markdown(
            f'<div class="metric-card green"><div class="metric-label">أفضل صفقة</div>'
            f'<div class="metric-value" style="font-size:18px;">{s["best_trade"]:+.2f}%</div></div>',
            unsafe_allow_html=True)
    with ia2:
        st.markdown(
            f'<div class="metric-card red"><div class="metric-label">أسوأ صفقة</div>'
            f'<div class="metric-value" style="font-size:18px;">{s["worst_trade"]:+.2f}%</div></div>',
            unsafe_allow_html=True)
    with ia3:
        avg = pnl_d / total if total > 0 else 0
        st.markdown(
            f'<div class="metric-card yellow"><div class="metric-label">متوسط الصفقة</div>'
            f'<div class="metric-value" style="font-size:18px;">{avg:+.3f}$</div></div>',
            unsafe_allow_html=True)
else:
    st.markdown(
        '<div style="text-align:center;padding:18px;color:#3a6a8a;font-family:Cairo,sans-serif;">'
        '⊘ لا توجد صفقات مكتملة بعد.</div>',
        unsafe_allow_html=True,
    )

# ── سجل الأحداث اللحظي ──
st.markdown('<div class="section-header">📟 سجل الأحداث اللحظي</div>', unsafe_allow_html=True)

# فلتر عرض السجلات
log_filter = st.selectbox(
    "عرض:",
    ["الكل","الصفقات فقط","الفحص فقط","التحذيرات"],
    label_visibility="collapsed",
)
filter_map = {
    "الكل":        None,
    "الصفقات فقط": ["buy","sell"],
    "الفحص فقط":   ["scan","pass","fail"],
    "التحذيرات":   ["warn"],
}
chosen = filter_map[log_filter]

logs_to_show = [
    l for l in st.session_state.logs
    if chosen is None or l["kind"] in chosen
]

if logs_to_show:
    log_html = '<div class="log-box">'
    for entry in logs_to_show[:40]:
        log_html += f'<div class="log-{entry["kind"]}">[{entry["ts"]}] {entry["msg"]}</div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="log-box" style="color:#3a6a8a;text-align:center;">'
        'في انتظار الأحداث...</div>',
        unsafe_allow_html=True,
    )

# ── حالة الذكاء الاصطناعي ──
st.markdown('<div class="section-header">🤖 حالة الذكاء الاصطناعي</div>', unsafe_allow_html=True)
a1, a2, a3, a4 = st.columns(4)
trained = st.session_state.ai_trained
n_hist  = len(st.session_state.ai_history)
progress = min(n_hist / 20 * 100, 100)

with a1:
    st.markdown(
        f'<div class="metric-card {"green" if trained else "yellow"}">'
        f'<div class="metric-label">حالة النموذج</div>'
        f'<div class="metric-value" style="font-size:16px;">{"✓ مُدرَّب" if trained else "⟳ يتعلم"}</div>'
        f'<div class="metric-sub">Gradient Boosting</div></div>',
        unsafe_allow_html=True,
    )
with a2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">عينات التدريب</div>'
        f'<div class="metric-value">{n_hist}</div>'
        f'<div class="metric-sub">يحتاج 20 للبدء ({progress:.0f}%)</div></div>',
        unsafe_allow_html=True,
    )
with a3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">حد الثقة للدخول</div>'
        f'<div class="metric-value">{MIN_AI_PROB*100:.0f}%</div>'
        f'<div class="metric-sub">أدنى نسبة مقبولة</div></div>',
        unsafe_allow_html=True,
    )
with a4:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">المؤشرات المستخدمة</div>'
        f'<div class="metric-value" style="font-size:14px;">6</div>'
        f'<div class="metric-sub">MFI،RSI،EMA،ATR،حجم،زخم</div></div>',
        unsafe_allow_html=True,
    )

# ── تذييل ──
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:11px;color:#1a3050;font-family:Cairo,sans-serif;">'
    '⚡ بوت المضاربة السريعة — وضع التداول الوهمي فقط — ليس نصيحة مالية ⚡'
    '</div>',
    unsafe_allow_html=True,
)

# التحديث التلقائي كل 8 ثوانٍ عند التشغيل
if st.session_state.running:
    time.sleep(8)
    st.rerun()
