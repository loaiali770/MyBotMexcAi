"""
╔══════════════════════════════════════════════════════════╗
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

# ── Scikit-learn ────────────────────────────────────────────
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

/* ── Base ── */
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

/* ── Metric Cards ── */
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

/* ── Section Headers ── */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 13px; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase;
    color: var(--accent); border-bottom: 1px solid var(--border);
    padding-bottom: 8px; margin: 20px 0 12px;
}

/* ── Tables ── */
.stDataFrame, [data-testid="stDataFrame"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
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

/* ── Inputs ── */
.stNumberInput input, .stTextInput input {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px !important;
}

/* ── Status Log ── */
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

/* ── Signal bars ── */
.sig-bar-wrap { display: flex; align-items: center; gap: 8px; }
.sig-bar {
    height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, var(--accent), var(--green));
}

/* ── Pulse dot ── */
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

/* ── Win/Loss badges ── */
.badge-win  { color: var(--green); font-weight: 700; }
.badge-loss { color: var(--red);   font-weight: 700; }
.badge-open { color: var(--yellow); font-weight: 700; }

/* ── Sidebar labels ── */
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
        "trades": [],          # list of trade dicts
        "scan_list": [],       # [{symbol, signal, mfi, volume_ratio, status}]
        "open_trades": {},     # symbol → trade dict
        "logs": deque(maxlen=60),
        "ai_model": None,
        "ai_scaler": None,
        "ai_trained": False,
        "ai_history": [],      # [(features, label)]
        "thread": None,
        "stop_event": threading.Event(),
        "stats": {"win":0,"loss":0,"pnl_today":0.0,"pnl_hour":0.0},
        "last_update": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════
MAX_OPEN = 2
TAKE_PROFIT_PCT = 0.015   # 1.5 %
INITIAL_SL_PCT  = 0.012   # 1.2 % initial stop
TRAILING_DIST   = 0.008   # 0.8 % trailing distance
MFI_BUY_THRESH  = 55      # MFI > 55 → money flowing in
VOL_MULTIPLIER  = 2.0     # volume must be 2× average
SCAN_INTERVAL   = 8       # seconds between scan cycles
MAX_PRICE       = 0.001   # only coins ≤ $0.001

WATCHLIST_BASE = [
    "PEPE/USDT","SHIB/USDT","FLOKI/USDT","BABYDOGE/USDT","BONK/USDT",
    "WIF/USDT","LUNC/USDT","XEC/USDT","HOT/USDT","BTT/USDT",
    "WIN/USDT","TRX/USDT","DOGE/USDT","SAMO/USDT","VOLT/USDT",
]

# ════════════════════════════════════════════════════════════
#  EXCHANGE
# ════════════════════════════════════════════════════════════
@st.cache_resource
def get_exchange():
    ex = ccxt.mexc({"enableRateLimit": True})
    return ex

exchange = get_exchange()

# ════════════════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════════════════
def compute_mfi(highs, lows, closes, volumes, period=14):
    tps = (np.array(highs) + np.array(lows) + np.array(closes)) / 3
    raw_mf = tps * np.array(volumes)
    pos, neg = np.zeros(len(tps)), np.zeros(len(tps))
    for i in range(1, len(tps)):
        if tps[i] > tps[i-1]:
            pos[i] = raw_mf[i]
        else:
            neg[i] = raw_mf[i]
    mfi = []
    for i in range(period, len(tps)):
        p = pos[i-period:i].sum()
        n = neg[i-period:i].sum()
        if n == 0:
            mfi.append(100.0)
        else:
            mfi.append(100 - 100/(1 + p/n))
    return mfi[-1] if mfi else 50.0

def compute_momentum(closes, period=5):
    if len(closes) < period+1:
        return 0.0
    return (closes[-1] - closes[-period-1]) / closes[-period-1] * 100

def compute_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i]-lows[i],
                 abs(highs[i]-closes[i-1]),
                 abs(lows[i]-closes[i-1]))
        trs.append(tr)
    return np.mean(trs[-period:]) if trs else closes[-1]*0.01

def compute_volume_ratio(volumes, lookback=20):
    if len(volumes) < lookback+1:
        return 1.0
    avg = np.mean(volumes[-lookback-1:-1])
    if avg == 0:
        return 1.0
    return volumes[-1] / avg

# ════════════════════════════════════════════════════════════
#  AI MODEL (Gradient Boosting — Online-style incremental)
# ════════════════════════════════════════════════════════════
def get_features(mfi, vol_ratio, momentum, atr_pct):
    return [mfi, vol_ratio, momentum, atr_pct]

def ai_predict(features):
    """Returns probability of a winning trade (0-1)."""
    m = st.session_state.ai_model
    sc = st.session_state.ai_scaler
    if m is None or not st.session_state.ai_trained:
        # No model yet — use heuristic score
        mfi, vol_ratio, momentum, atr_pct = features
        score = 0.0
        if mfi > MFI_BUY_THRESH:      score += 0.35
        if vol_ratio > VOL_MULTIPLIER: score += 0.35
        if momentum > 0.3:             score += 0.20
        if atr_pct < 0.05:             score += 0.10
        return min(score, 0.99)
    try:
        X = sc.transform([features])
        prob = m.predict_proba(X)[0][1]
        return float(prob)
    except Exception:
        return 0.5

def ai_retrain():
    hist = st.session_state.ai_history
    if len(hist) < 20:
        return
    X = [h[0] for h in hist]
    y = [h[1] for h in hist]
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3,
        learning_rate=0.1, subsample=0.8,
        random_state=42
    )
    try:
        model.fit(X_scaled, y)
        st.session_state.ai_model = model
        st.session_state.ai_scaler = sc
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
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        return df
    except Exception:
        return None

def fetch_ticker(symbol):
    try:
        return exchange.fetch_ticker(symbol)
    except Exception:
        return None

# ════════════════════════════════════════════════════════════
#  CORE BOT LOGIC
# ════════════════════════════════════════════════════════════
def open_trade(symbol, price, atr, features):
    sl = price - INITIAL_SL_PCT * price
    tp = price + TAKE_PROFIT_PCT * price
    alloc = st.session_state.balance / (MAX_OPEN - len(st.session_state.open_trades))
    qty = alloc / price
    trade = {
        "symbol": symbol,
        "entry_price": price,
        "qty": qty,
        "sl": sl,
        "tp": tp,
        "trail_high": price,
        "trail_sl": sl,
        "entry_time": datetime.now().isoformat(timespec="seconds"),
        "exit_price": None,
        "exit_time": None,
        "pnl_pct": None,
        "status": "OPEN",
        "features": features,
        "atr": atr,
    }
    st.session_state.open_trades[symbol] = trade
    st.session_state.balance -= alloc
    add_log(f"BUY  {symbol}  @ ${price:.8f}  SL=${sl:.8f}  TP=${tp:.8f}", "buy")

def close_trade(symbol, price, reason=""):
    trade = st.session_state.open_trades.pop(symbol, None)
    if not trade:
        return
    pnl_pct = (price - trade["entry_price"]) / trade["entry_price"] * 100
    pnl_usdt = (price - trade["entry_price"]) * trade["qty"]
    st.session_state.balance += trade["qty"] * price
    trade["exit_price"] = price
    trade["exit_time"] = datetime.now().isoformat(timespec="seconds")
    trade["pnl_pct"] = round(pnl_pct, 3)
    trade["status"] = "WIN" if pnl_pct > 0 else "LOSS"
    st.session_state.trades.append(trade)
    # stats
    st.session_state.stats["pnl_today"]  = round(st.session_state.stats["pnl_today"] + pnl_usdt, 4)
    st.session_state.stats["pnl_hour"]   = round(st.session_state.stats["pnl_hour"]  + pnl_usdt, 4)
    if pnl_pct > 0:
        st.session_state.stats["win"] += 1
        add_log(f"SELL {symbol} @ ${price:.8f}  PnL={pnl_pct:+.2f}%  [{reason}]", "buy")
    else:
        st.session_state.stats["loss"] += 1
        add_log(f"SELL {symbol} @ ${price:.8f}  PnL={pnl_pct:+.2f}%  [{reason}]", "sell")
    # teach AI
    ai_record(trade["features"], pnl_pct > 0)

def manage_open_trades():
    """Update trailing SL and check TP/SL hits."""
    for sym in list(st.session_state.open_trades.keys()):
        t = st.session_state.open_trades[sym]
        ticker = fetch_ticker(sym)
        if not ticker:
            continue
        price = ticker["last"]
        # Update trailing stop
        if price > t["trail_high"]:
            t["trail_high"] = price
            t["trail_sl"]   = price * (1 - TRAILING_DIST)
        # Check exits
        if price >= t["tp"]:
            close_trade(sym, price, "TP")
        elif price <= t["trail_sl"]:
            close_trade(sym, price, "Trail-SL")
        elif price <= t["sl"]:
            close_trade(sym, price, "SL")

def scan_symbols():
    """Scan watchlist for entry signals."""
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
            mfi        = compute_mfi(highs, lows, closes, volumes)
            vol_ratio  = compute_volume_ratio(volumes)
            momentum   = compute_momentum(closes)
            atr        = compute_atr(highs, lows, closes)
            atr_pct    = atr / closes[-1] if closes[-1] > 0 else 0.01
            features   = get_features(mfi, vol_ratio, momentum, atr_pct)
            ai_prob    = ai_predict(features)
            results.append({
                "symbol":     sym,
                "price":      price,
                "mfi":        round(mfi, 1),
                "vol_ratio":  round(vol_ratio, 2),
                "momentum":   round(momentum, 3),
                "ai_prob":    round(ai_prob * 100, 1),
                "signal":     "⚡ STRONG" if ai_prob > 0.65 else ("◎ WATCH" if ai_prob > 0.45 else "○ WEAK"),
                "features":   features,
                "atr":        atr,
                "current_price": price,
            })
            # Entry decision
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
    st.session_state.scan_list = results
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
    st.session_state.balance = st.session_state.budget
    st.session_state.trades = []
    st.session_state.open_trades = {}
    st.session_state.stats = {"win":0,"loss":0,"pnl_today":0.0,"pnl_hour":0.0}
    st.session_state.logs = deque(maxlen=60)
    t = threading.Thread(target=bot_loop, args=(st.session_state.stop_event,), daemon=True)
    t.start()
    st.session_state.thread = t
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
    running = st.session_state.running
    dot_cls = "pulse" if running else "pulse stopped"
    status_txt = "RUNNING" if running else "STOPPED"
    st.markdown(
        f'<div style="text-align:center;margin-bottom:16px;">'
        f'<span class="{dot_cls}"></span>'
        f'<span style="font-family:Rajdhani,sans-serif;font-size:14px;letter-spacing:2px;">{status_txt}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    budget = st.number_input("Budget (USDT)", min_value=10.0, max_value=10000.0,
                              value=float(st.session_state.budget), step=10.0, key="budget_input")
    st.session_state.budget = budget

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        if st.button("▶ START", use_container_width=True):
            start_bot()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        if st.button("■ STOP", use_container_width=True):
            stop_bot()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="metric-label">Strategy</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#5a7a99;line-height:1.8;">
    📊 Volume Breakout + MFI<br>
    🤖 Gradient Boosting AI<br>
    🎯 TP: 1.5% | Init SL: 1.2%<br>
    🔄 Trailing Stop: 0.8%<br>
    💰 Max Price: $0.001<br>
    ⚡ Max Open: 2 Trades
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    bal = st.session_state.balance
    init = st.session_state.budget
    bal_pct = (bal - init) / init * 100 if init > 0 else 0
    bal_color = "#00ff9d" if bal >= init else "#ff3860"
    st.markdown(
        f'<div class="metric-label">Current Balance</div>'
        f'<div style="font-size:22px;font-weight:700;color:{bal_color};">${bal:.2f}</div>'
        f'<div style="font-size:11px;color:{bal_color};">{bal_pct:+.2f}% from start</div>',
        unsafe_allow_html=True
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
    unsafe_allow_html=True
)

# ── TOP METRICS ────────────────────────────────────────────
stats = st.session_state.stats
total = stats["win"] + stats["loss"]
wr = stats["win"] / total * 100 if total > 0 else 0.0
pnl_d = stats["pnl_today"]
pnl_h = stats["pnl_hour"]

c1, c2, c3, c4, c5 = st.columns(5)
def metric_card(col, label, value, sub="", css_class=""):
    with col:
        st.markdown(
            f'<div class="metric-card {css_class}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

metric_card(c1, "PnL Today",
            f'{"+" if pnl_d>=0 else ""}{pnl_d:.2f} USDT',
            "Paper trading",
            "green" if pnl_d >= 0 else "red")
metric_card(c2, "PnL This Hour",
            f'{"+" if pnl_h>=0 else ""}{pnl_h:.2f} USDT',
            "Rolling",
            "green" if pnl_h >= 0 else "red")
metric_card(c3, "Win / Loss",
            f'{stats["win"]} / {stats["loss"]}',
            f'{total} total trades',
            "yellow")
metric_card(c4, "Win Rate",
            f'{wr:.1f}%',
            "AI learning..." if not st.session_state.ai_trained else "AI active",
            "green" if wr >= 50 else "red")
metric_card(c5, "Open Trades",
            f'{len(st.session_state.open_trades)} / {MAX_OPEN}',
            "max 2 simultaneous",
            "")

# ── OPEN TRADES ────────────────────────────────────────────
st.markdown('<div class="section-header">🔴 LIVE OPEN POSITIONS</div>', unsafe_allow_html=True)
if st.session_state.open_trades:
    rows = []
    for sym, t in st.session_state.open_trades.items():
        ticker = fetch_ticker(sym)
        cur = ticker["last"] if ticker else t["entry_price"]
        pnl_pct = (cur - t["entry_price"]) / t["entry_price"] * 100
        rows.append({
            "Symbol":   sym,
            "Entry $":  f'{t["entry_price"]:.8f}',
            "Current $": f'{cur:.8f}',
            "TP $":     f'{t["tp"]:.8f}',
            "Trail SL $": f'{t["trail_sl"]:.8f}',
            "PnL %":    f'{pnl_pct:+.3f}%',
            "Status":   "🟢 OPEN",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div style="text-align:center;padding:20px;color:#4a6280;font-size:13px;">'
        '⊘ No open positions — bot is scanning for entries...</div>',
        unsafe_allow_html=True
    )

# ── SCANNING LIST ──────────────────────────────────────────
st.markdown('<div class="section-header">🔍 AI SCANNING LIST</div>', unsafe_allow_html=True)
scan = st.session_state.scan_list
if scan:
    scan_rows = []
    for s in scan:
        bar_w = int(s["ai_prob"])
        scan_rows.append({
            "Symbol":    s["symbol"],
            "Price $":   f'{s["price"]:.8f}' if s.get("price") else "—",
            "MFI":       s["mfi"],
            "Vol Ratio": f'×{s["vol_ratio"]}',
            "Momentum":  f'{s["momentum"]:+.3f}%',
            "AI Prob":   f'{s["ai_prob"]}%',
            "Signal":    s["signal"],
        })
    st.dataframe(pd.DataFrame(scan_rows), use_container_width=True, hide_index=True)
    ts = st.session_state.last_update
    st.markdown(
        f'<div style="font-size:10px;color:#4a6280;text-align:right;margin-top:4px;">'
        f'Last scan: {ts}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div style="text-align:center;padding:20px;color:#4a6280;font-size:13px;">'
        '⊘ Start bot to begin scanning...</div>',
        unsafe_allow_html=True
    )

# ── TRADE HISTORY ──────────────────────────────────────────
st.markdown('<div class="section-header">📋 TRADE HISTORY</div>', unsafe_allow_html=True)
trades = st.session_state.trades
if trades:
    hist_rows = []
    for t in reversed(trades[-50:]):
        pnl = t.get("pnl_pct", 0) or 0
        badge = "🟢 WIN" if t["status"] == "WIN" else "🔴 LOSS"
        hist_rows.append({
            "Symbol":      t["symbol"],
            "Entry Time":  t["entry_time"],
            "Entry $":     f'{t["entry_price"]:.8f}',
            "Exit $":      f'{t["exit_price"]:.8f}' if t["exit_price"] else "—",
            "PnL %":       f'{pnl:+.3f}%',
            "Status":      badge,
        })
    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div style="text-align:center;padding:20px;color:#4a6280;font-size:13px;">'
        '⊘ No completed trades yet.</div>',
        unsafe_allow_html=True
    )

# ── STATUS LOGS ────────────────────────────────────────────
st.markdown('<div class="section-header">📡 STATUS LOGS</div>', unsafe_allow_html=True)
if st.session_state.logs:
    log_html = '<div class="log-box">'
    for entry in st.session_state.logs:
        cls = f'log-{entry["kind"]}'
        log_html += f'<div class="{cls}">[{entry["ts"]}] {entry["msg"]}</div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="log-box" style="color:#4a6280;text-align:center;">'
        'Waiting for activity...</div>',
        unsafe_allow_html=True
    )

# ── AI STATUS ──────────────────────────────────────────────
st.markdown('<div class="section-header">🤖 AI MODEL STATUS</div>', unsafe_allow_html=True)
ai_col1, ai_col2, ai_col3 = st.columns(3)
with ai_col1:
    trained = st.session_state.ai_trained
    st.markdown(
        f'<div class="metric-card {"green" if trained else "yellow"}">'
        f'<div class="metric-label">Model Status</div>'
        f'<div class="metric-value" style="font-size:16px;">{"✓ TRAINED" if trained else "○ LEARNING"}</div>'
        f'<div class="metric-sub">GradientBoosting</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with ai_col2:
    n_hist = len(st.session_state.ai_history)
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Training Samples</div>'
        f'<div class="metric-value">{n_hist}</div>'
        f'<div class="metric-sub">Need 20 to train</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with ai_col3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Entry Threshold</div>'
        f'<div class="metric-value">60%</div>'
        f'<div class="metric-sub">AI confidence min</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── FOOTER ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:10px;color:#2a4060;letter-spacing:2px;">'
    '⚡ MEXC SCALPING BOT — PAPER TRADING ONLY — NOT FINANCIAL ADVICE ⚡'
    '</div>',
    unsafe_allow_html=True
)

# Auto-refresh every 10s when bot is running
if st.session_state.running:
    time.sleep(10)
    st.rerun()
