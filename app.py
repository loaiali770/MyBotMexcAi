"""
╔══════════════════════════════════════════════════════════════╗
║        بوت المضاربة السريعة — MEXC Scalping Bot v3.0        ║
║  فحص تلقائي لجميع العملات + Compound + TP ديناميكي         ║
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
    page_title="بوت المضاربة v3",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Cairo:wght@400;600;700;900&display=swap');
:root {
    --bg:      #060b10;
    --card:    #0b1520;
    --panel:   #0f1e2e;
    --border:  #1a3050;
    --accent:  #00c8ff;
    --green:   #00ff9d;
    --red:     #ff3860;
    --yellow:  #ffd600;
    --orange:  #ff8c00;
    --purple:  #c084fc;
    --text:    #c8ddf0;
    --dim:     #4a6a8a;
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
.b-hold{background:#1a1000;color:var(--yellow);border:1px solid var(--yellow);}

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
.log-compound{color:var(--orange);}
.log-withdraw{color:var(--yellow);}

.pulse{display:inline-block;width:9px;height:9px;border-radius:50%;
    background:var(--green);animation:blink 1.2s infinite;margin-left:6px;}
@keyframes blink{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.6)}}
.pulse.off{background:var(--red);animation:none;}

.compound-banner{
    background:linear-gradient(135deg,#0a2010,#1a3020);
    border:1px solid var(--green);border-radius:12px;
    padding:16px 20px;margin-bottom:12px;
    font-family:'Cairo',sans-serif;
}
.withdraw-banner{
    background:linear-gradient(135deg,#2a1a00,#3a2a00);
    border:2px solid var(--yellow);border-radius:12px;
    padding:16px 20px;margin-bottom:12px;
    font-family:'Cairo',sans-serif;
    animation:glow 1s infinite alternate;
}
@keyframes glow{
    from{box-shadow:0 0 5px var(--yellow);}
    to{box-shadow:0 0 20px var(--yellow),0 0 40px #ff8c0050;}
}

.stButton>button{
    font-family:'Cairo',sans-serif!important;font-weight:700!important;
    font-size:14px!important;border-radius:8px!important;
    border:1px solid var(--accent)!important;
    color:var(--accent)!important;background:transparent!important;
    transition:all .2s!important;width:100%;
}
.stButton>button:hover{background:var(--accent)!important;color:var(--bg)!important;}
.sb>button{border-color:var(--green)!important;color:var(--green)!important;}
.sb>button:hover{background:var(--green)!important;color:var(--bg)!important;}
.xb>button{border-color:var(--red)!important;color:var(--red)!important;}
.xb>button:hover{background:var(--red)!important;color:var(--bg)!important;}
.wb>button{border-color:var(--yellow)!important;color:var(--yellow)!important;}
.wb>button:hover{background:var(--yellow)!important;color:var(--bg)!important;}

.prog-wrap{background:#0a1a2a;border-radius:6px;height:10px;overflow:hidden;margin-top:6px;}
.prog-bar{height:100%;border-radius:6px;
    background:linear-gradient(90deg,var(--accent),var(--green));
    transition:width .5s;}

[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:10px!important;}
.stNumberInput input,.stSelectbox select{
    background:var(--panel)!important;border:1px solid var(--border)!important;
    color:var(--text)!important;font-family:'JetBrains Mono',monospace!important;
    border-radius:6px!important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  تهيئة الحالة
# ════════════════════════════════════════════════════════════
def init_state():
    d = {
        "running":         False,
        "budget":          20.0,
        "balance":         20.0,
        "initial_capital": 20.0,
        "withdrawn":       0.0,
        "compound_log":    [],
        "withdraw_ready":  False,
        "trades":          [],
        "scan_list":       [],
        "open_trades":     {},
        "logs":            deque(maxlen=120),
        "scan_results":    deque(maxlen=50),
        "alerts":          deque(maxlen=5),
        "ai_model":        None,
        "ai_scaler":       None,
        "ai_trained":      False,
        "ai_history":      [],
        "stop_event":      threading.Event(),
        "current_scan":    "",
        "all_symbols":     [],
        "symbols_loaded":  False,
        "stats": {
            "win":0,"loss":0,"pnl_today":0.0,"pnl_total":0.0,
            "total_fees":0.0,"best_trade":0.0,"worst_trade":0.0,
            "moon_shots":0,"total_scanned":0,
        },
        "last_update": None,
        "scan_cycle":  0,
    }
    for k,v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ════════════════════════════════════════════════════════════
#  الثوابت
# ════════════════════════════════════════════════════════════
MAX_OPEN          = 3       # زيادة للاستفادة من الفرص
MAX_PRICE         = 0.001
COMMISSION        = 0.002   # 0.2% لكل اتجاه
INITIAL_SL_PCT    = 0.012
TRAILING_DIST     = 0.007
MFI_THRESH        = 55
VOL_MULT          = 2.0
MIN_AI_PROB       = 0.60
BASE_TP           = 0.018   # 1.8% هدف أساسي
MOON_TP           = 999.0   # لا سقف للربح في الصفقات القوية
MOON_THRESHOLD    = 0.82    # إذا AI > 82% → وضع القمر 🌙
REVERSAL_DROP     = 0.008   # إذا ارتد 0.8% من القمة → اخرج
SCAN_DELAY        = 0.3     # ثانية بين كل عملة (لتجنب الحظر)
COMPOUND_NOTIFY   = 1.0     # عندما يتضاعف رأس المال أخبر الأدمن

# ════════════════════════════════════════════════════════════
#  الاتصال بالبورصة
# ════════════════════════════════════════════════════════════
@st.cache_resource
def get_exchange():
    return ccxt.mexc({"enableRateLimit": True, "timeout": 15000})

exchange = get_exchange()

# ════════════════════════════════════════════════════════════
#  جلب جميع العملات الرخيصة تلقائياً
# ════════════════════════════════════════════════════════════
def load_all_cheap_symbols():
    """يجلب جميع عملات MEXC التي سعرها ≤ $0.001"""
    try:
        tickers = exchange.fetch_tickers()
        cheap = []
        for sym, t in tickers.items():
            if not sym.endswith("/USDT"):
                continue
            price = t.get("last") or 0
            vol   = t.get("quoteVolume") or 0
            if 0 < price <= MAX_PRICE and vol > 1000:  # حجم تداول لا يقل عن $1000
                cheap.append({"symbol": sym, "price": price, "volume": vol})
        # ترتيب حسب حجم التداول الأعلى أولاً
        cheap.sort(key=lambda x: x["volume"], reverse=True)
        symbols = [c["symbol"] for c in cheap]
        add_log(f"✓ تم تحميل {len(symbols)} عملة رخيصة من MEXC", "info")
        return symbols
    except Exception as e:
        add_log(f"⚠ خطأ في تحميل العملات: {e}", "warn")
        return []

# ════════════════════════════════════════════════════════════
#  المؤشرات الفنية
# ════════════════════════════════════════════════════════════
def compute_mfi(highs, lows, closes, volumes, period=14):
    tps = (np.array(highs)+np.array(lows)+np.array(closes))/3
    mf  = tps * np.array(volumes)
    pos = np.zeros(len(tps))
    neg = np.zeros(len(tps))
    for i in range(1,len(tps)):
        (pos if tps[i]>tps[i-1] else neg)[i] = mf[i]
    vals = []
    for i in range(period,len(tps)):
        p = pos[i-period:i].sum()
        n = neg[i-period:i].sum()
        vals.append(100.0 if n==0 else 100-100/(1+p/n))
    return vals[-1] if vals else 50.0

def compute_rsi(closes, period=14):
    if len(closes) < period+1: return 50.0
    d = np.diff(closes)
    g = np.where(d>0,d,0)
    l = np.where(d<0,-d,0)
    ag = np.mean(g[-period:])
    al = np.mean(l[-period:])
    return 100.0 if al==0 else 100-100/(1+ag/al)

def compute_momentum(closes, period=5):
    if len(closes)<period+1: return 0.0
    return (closes[-1]-closes[-period-1])/closes[-period-1]*100

def compute_atr(highs, lows, closes, period=14):
    trs = [max(highs[i]-lows[i],
               abs(highs[i]-closes[i-1]),
               abs(lows[i]-closes[i-1]))
           for i in range(1,len(closes))]
    return np.mean(trs[-period:]) if trs else closes[-1]*0.01

def compute_volume_ratio(volumes, lookback=20):
    if len(volumes)<lookback+1: return 1.0
    avg = np.mean(volumes[-lookback-1:-1])
    return 1.0 if avg==0 else volumes[-1]/avg

def compute_ema(closes, period=20):
    if len(closes)<period: return closes[-1]
    k = 2/(period+1)
    e = closes[0]
    for c in closes[1:]: e = c*k+e*(1-k)
    return e

def detect_reversal(closes, trail_high, current_price):
    """كشف الارتداد — إذا انخفض السعر عن القمة بنسبة معينة"""
    if trail_high <= 0: return False
    drop = (trail_high - current_price) / trail_high
    # أيضاً نفحص إذا كان الزخم الأخير سلبياً
    momentum = compute_momentum(closes[-6:], 3) if len(closes)>=6 else 0
    return drop >= REVERSAL_DROP and momentum < -0.2

# ════════════════════════════════════════════════════════════
#  الذكاء الاصطناعي المحسّن
# ════════════════════════════════════════════════════════════
def get_features(mfi, vol_ratio, momentum, atr_pct, rsi, ema_dist, vol_trend):
    return [mfi, vol_ratio, momentum, atr_pct, rsi, ema_dist, vol_trend]

def ai_predict(features):
    m  = st.session_state.ai_model
    sc = st.session_state.ai_scaler
    if m is None or not st.session_state.ai_trained:
        mfi,vol_ratio,momentum,atr_pct,rsi,ema_dist,vol_trend = features
        s = 0.0
        if mfi > MFI_THRESH:       s += 0.25
        if vol_ratio > VOL_MULT:   s += 0.25
        if momentum > 0.3:         s += 0.15
        if rsi > 50 and rsi < 78:  s += 0.15
        if ema_dist > 0:           s += 0.10
        if vol_trend > 1.5:        s += 0.10
        return min(s, 0.99)
    try:
        X = sc.transform([features])
        return float(m.predict_proba(X)[0][1])
    except:
        return 0.5

def ai_retrain():
    hist = st.session_state.ai_history
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
        st.session_state.ai_model   = model
        st.session_state.ai_scaler  = sc
        st.session_state.ai_trained = True
        n = len(hist)
        add_log(f"🧠 تم تدريب الذكاء الاصطناعي على {n} صفقة", "info")
    except:
        pass

def ai_record(features, won):
    st.session_state.ai_history.append((features, int(won)))
    if len(st.session_state.ai_history) % 10 == 0:
        ai_retrain()

# ════════════════════════════════════════════════════════════
#  السجلات
# ════════════════════════════════════════════════════════════
def add_log(msg, kind="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.appendleft({"ts":ts,"msg":msg,"kind":kind})

def add_scan_result(symbol, status, reason, ai_prob=0, price=0):
    st.session_state.scan_results.appendleft({
        "symbol":  symbol,
        "status":  status,
        "reason":  reason,
        "ai_prob": ai_prob,
        "price":   price,
        "time":    datetime.now().strftime("%H:%M:%S"),
    })

def add_alert(msg):
    st.session_state.alerts.appendleft({"msg":msg,"time":datetime.now().strftime("%H:%M:%S")})

# ════════════════════════════════════════════════════════════
#  Compound — إعادة استثمار الأرباح
# ════════════════════════════════════════════════════════════
def check_compound():
    """يتحقق إذا تضاعف رأس المال → يُخطر الأدمن"""
    bal  = st.session_state.balance
    init = st.session_state.initial_capital
    withdrawn = st.session_state.withdrawn

    # حساب الربح الكلي
    total_profit = (bal + withdrawn) - init

    # إذا الربح = رأس المال الأصلي → وقت الإخطار
    if total_profit >= init and not st.session_state.withdraw_ready:
        st.session_state.withdraw_ready = True
        msg = (f"🎉 رأس المال تضاعف! "
               f"البداية: ${init:.2f} | الرصيد: ${bal:.2f} | "
               f"الربح: ${total_profit:.2f}")
        add_log(msg, "withdraw")
        add_alert(f"💰 الربح وصل ${total_profit:.2f} — هل تريد سحب الأرباح؟")
        st.session_state.compound_log.append({
            "time":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "event":  "تضاعف رأس المال",
            "balance": bal,
            "profit":  total_profit,
        })

def do_withdraw():
    """سحب الأرباح والإبقاء على رأس المال الأصلي فقط"""
    bal  = st.session_state.balance
    init = st.session_state.initial_capital
    if bal <= init: return
    profit = bal - init
    st.session_state.withdrawn += profit
    st.session_state.balance    = init
    st.session_state.withdraw_ready = False
    add_log(f"💸 تم سحب ${profit:.4f} — الرصيد المتداول: ${init:.2f}", "withdraw")
    add_alert(f"✅ تم سحب الأرباح: ${profit:.4f}")

# ════════════════════════════════════════════════════════════
#  منطق التداول
# ════════════════════════════════════════════════════════════
def open_trade(symbol, price, atr, features, is_moon=False):
    slots = MAX_OPEN - len(st.session_state.open_trades)
    if slots <= 0: return
    # توزيع رأس المال — نستخدم حصة من الرصيد
    alloc     = st.session_state.balance / MAX_OPEN
    fee_in    = alloc * COMMISSION
    qty       = (alloc - fee_in) / price
    sl        = price * (1 - INITIAL_SL_PCT)
    # هدف الربح ديناميكي
    tp        = price * (1 + (MOON_TP if is_moon else BASE_TP))

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
        "entry_time":  datetime.now().isoformat(timespec="seconds"),
        "exit_price":  None,
        "exit_time":   None,
        "pnl_pct":     None,
        "status":      "OPEN",
        "features":    features,
        "peak_gain":   0.0,
    }
    st.session_state.open_trades[symbol] = trade
    st.session_state.balance -= alloc
    st.session_state.stats["total_fees"] += fee_in

    mode = "🌙 وضع القمر" if is_moon else "⚡ عادي"
    add_log(f"▲ شراء {symbol} @ ${price:.8f} | {mode} | SL=${sl:.8f}", "buy")
    add_alert(f"✅ شراء: {symbol} @ ${price:.8f} [{mode}]")

def close_trade(symbol, price, reason=""):
    trade = st.session_state.open_trades.pop(symbol, None)
    if not trade: return

    fee_out   = trade["qty"] * price * COMMISSION
    gross     = (price - trade["entry_price"]) * trade["qty"]
    net_usdt  = gross - trade["fee_in"] - fee_out
    pnl_pct   = net_usdt / trade["alloc"] * 100

    # إعادة الرصيد مع الربح/الخسارة (Compound تلقائي)
    st.session_state.balance += trade["qty"] * price - fee_out
    st.session_state.stats["total_fees"] += fee_out

    trade.update({
        "exit_price": price,
        "exit_time":  datetime.now().isoformat(timespec="seconds"),
        "pnl_pct":    round(pnl_pct, 3),
        "pnl_net":    round(net_usdt, 6),
        "status":     "WIN" if pnl_pct > 0 else "LOSS",
    })
    st.session_state.trades.append(trade)

    s = st.session_state.stats
    s["pnl_today"] = round(s["pnl_today"] + net_usdt, 6)
    s["pnl_total"] = round(s["pnl_total"] + net_usdt, 6)

    if pnl_pct > 0:
        s["win"] += 1
        s["best_trade"] = max(s["best_trade"], pnl_pct)
        if pnl_pct > 10:
            s["moon_shots"] += 1
            add_log(f"🌙 مكسب كبير! {symbol} {pnl_pct:+.1f}% [{reason}]", "moon")
        else:
            add_log(f"▼ بيع {symbol} | ربح: {pnl_pct:+.2f}% [{reason}]", "buy")
        add_alert(f"💰 ربح: {symbol} {pnl_pct:+.2f}%")
    else:
        s["loss"] += 1
        s["worst_trade"] = min(s["worst_trade"], pnl_pct)
        add_log(f"▼ بيع {symbol} | خسارة: {pnl_pct:+.2f}% [{reason}]", "sell")
        add_alert(f"⛔ خسارة: {symbol} {pnl_pct:+.2f}%")

    ai_record(trade["features"], pnl_pct > 0)

    # تحقق من Compound بعد كل صفقة
    check_compound()

def manage_open_trades():
    for sym in list(st.session_state.open_trades.keys()):
        t      = st.session_state.open_trades[sym]
        ticker = fetch_ticker(sym)
        if not ticker: continue
        price = ticker["last"]

        # تحديث أعلى سعر وأعلى ربح
        if price > t["trail_high"]:
            t["trail_high"] = price
            t["trail_sl"]   = price * (1 - TRAILING_DIST)

        peak_gain = (t["trail_high"] - t["entry_price"]) / t["entry_price"] * 100
        t["peak_gain"] = peak_gain

        # ── وضع القمر 🌙: لا نخرج إلا عند الارتداد ──
        if t["is_moon"]:
            # جلب بيانات الشمعات لكشف الارتداد
            df = fetch_ohlcv(sym, limit=15)
            closes = df["close"].tolist() if df is not None else [price]
            if detect_reversal(closes, t["trail_high"], price):
                close_trade(sym, price, f"ارتداد من القمة ({peak_gain:+.1f}%)")
            elif price <= t["sl"]:
                close_trade(sym, price, "وقف الخسارة")
        else:
            # وضع عادي
            if price >= t["tp"]:
                close_trade(sym, price, "هدف الربح ✓")
            elif price <= t["trail_sl"]:
                close_trade(sym, price, "وقف متحرك")
            elif price <= t["sl"]:
                close_trade(sym, price, "وقف الخسارة")

def fetch_ohlcv(symbol, timeframe="1m", limit=60):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not data or len(data) < 20: return None
        return pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    except:
        return None

def fetch_ticker(symbol):
    try:
        return exchange.fetch_ticker(symbol)
    except:
        return None

# ════════════════════════════════════════════════════════════
#  محرك الفحص الشامل — كل العملات الرخيصة
# ════════════════════════════════════════════════════════════
def scan_symbols():
    symbols = st.session_state.all_symbols
    if not symbols: return

    st.session_state.scan_cycle += 1
    results = []

    for sym in symbols:
        if st.session_state.stop_event.is_set(): break

        st.session_state.current_scan = sym
        st.session_state.stats["total_scanned"] += 1

        try:
            # تحقق من الحد الأقصى
            if (len(st.session_state.open_trades) >= MAX_OPEN
                    and sym not in st.session_state.open_trades):
                time.sleep(SCAN_DELAY * 0.1)
                continue

            ticker = fetch_ticker(sym)
            if not ticker:
                time.sleep(SCAN_DELAY)
                continue

            price = ticker.get("last") or 0
            if not price or price > MAX_PRICE:
                time.sleep(SCAN_DELAY * 0.2)
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
            ema_dist  = (closes[-1] - ema20) / ema20 * 100
            # اتجاه الحجم: هل الحجم الأخير أكبر من السابق؟
            vol_trend = volumes[-1] / volumes[-2] if len(volumes)>1 and volumes[-2]>0 else 1.0

            features  = get_features(mfi, vol_ratio, momentum, atr_pct, rsi, ema_dist, vol_trend)
            ai_prob   = ai_predict(features)

            # تحديد وضع القمر
            is_moon = ai_prob >= MOON_THRESHOLD and vol_ratio > 3.0 and mfi > 65

            # تصنيف الإشارة
            if is_moon:
                signal = "🌙 قمر"
            elif ai_prob > 0.70:
                signal = "⚡ قوي جداً"
            elif ai_prob > 0.60:
                signal = "◎ جيد"
            elif ai_prob > 0.45:
                signal = "○ متابعة"
            else:
                signal = "✗ ضعيف"

            results.append({
                "symbol":    sym,
                "price":     price,
                "mfi":       round(mfi, 1),
                "vol_ratio": round(vol_ratio, 2),
                "momentum":  round(momentum, 3),
                "ai_prob":   round(ai_prob * 100, 1),
                "signal":    signal,
                "is_moon":   is_moon,
                "features":  features,
                "atr":       atr,
            })

            # ── قرار الدخول ──
            can_enter = (
                sym not in st.session_state.open_trades
                and len(st.session_state.open_trades) < MAX_OPEN
                and st.session_state.balance > 1.0
                and mfi > MFI_THRESH
                and vol_ratio > VOL_MULT
                and ai_prob >= MIN_AI_PROB
            )

            if can_enter:
                reason_ok = (
                    f"MFI={mfi:.0f} | Vol=×{vol_ratio:.1f} | "
                    f"AI={ai_prob*100:.0f}% | {'🌙 وضع القمر' if is_moon else '⚡ عادي'}"
                )
                add_scan_result(sym, "moon" if is_moon else "pass", reason_ok, ai_prob, price)
                open_trade(sym, price, atr, features, is_moon)
            else:
                # سبب الرفض
                if mfi <= MFI_THRESH:
                    reason = f"MFI منخفض ({mfi:.0f})"
                elif vol_ratio <= VOL_MULT:
                    reason = f"حجم ضعيف (×{vol_ratio:.1f})"
                elif ai_prob < MIN_AI_PROB:
                    reason = f"AI منخفض ({ai_prob*100:.0f}%)"
                else:
                    reason = "صفقات مفتوحة / رصيد"
                add_scan_result(sym, "fail", reason, ai_prob, price)

        except Exception as e:
            pass

        time.sleep(SCAN_DELAY)

    results.sort(key=lambda x: x["ai_prob"], reverse=True)
    st.session_state.scan_list   = results[:30]
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
    st.session_state.current_scan = ""

# ════════════════════════════════════════════════════════════
#  خيط البوت — يعمل 24/7
# ════════════════════════════════════════════════════════════
def bot_loop(stop_event):
    add_log("⚡ البوت يعمل 24/7 — جاري تحميل العملات...", "info")

    # تحميل العملات مرة واحدة
    symbols = load_all_cheap_symbols()
    st.session_state.all_symbols    = symbols
    st.session_state.symbols_loaded = True
    add_log(f"📋 قائمة الفحص: {len(symbols)} عملة", "info")

    while not stop_event.is_set():
        try:
            # إدارة الصفقات المفتوحة أولاً
            manage_open_trades()
            # ثم فحص شامل لكل العملات
            scan_symbols()
            # تحديث قائمة العملات كل ساعة
            if st.session_state.scan_cycle % 60 == 0:
                symbols = load_all_cheap_symbols()
                st.session_state.all_symbols = symbols
        except Exception as e:
            add_log(f"⚠ خطأ: {e}", "warn")
            time.sleep(5)

    add_log("■ توقف البوت.", "warn")

def start_bot():
    if st.session_state.running: return
    st.session_state.stop_event.clear()
    st.session_state.initial_capital = st.session_state.budget
    st.session_state.balance         = st.session_state.budget
    st.session_state.withdrawn       = 0.0
    st.session_state.withdraw_ready  = False
    st.session_state.trades          = []
    st.session_state.open_trades     = {}
    st.session_state.scan_results    = deque(maxlen=50)
    st.session_state.logs            = deque(maxlen=120)
    st.session_state.alerts          = deque(maxlen=5)
    st.session_state.scan_cycle      = 0
    st.session_state.stats = {
        "win":0,"loss":0,"pnl_today":0.0,"pnl_total":0.0,
        "total_fees":0.0,"best_trade":0.0,"worst_trade":0.0,
        "moon_shots":0,"total_scanned":0,
    }
    t = threading.Thread(target=bot_loop, args=(st.session_state.stop_event,), daemon=True)
    t.start()
    st.session_state.running = True

def stop_bot():
    if not st.session_state.running: return
    st.session_state.stop_event.set()
    st.session_state.running = False

# ════════════════════════════════════════════════════════════
#  الشريط الجانبي
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 16px;border-bottom:1px solid var(--border);margin-bottom:16px;">
    <div style="font-family:Cairo;font-size:20px;font-weight:900;color:#00c8ff;">⚡ بوت المضاربة</div>
    <div style="font-size:11px;color:#4a6a8a;">MEXC Scalping Bot v3.0</div>
    </div>""", unsafe_allow_html=True)

    running = st.session_state.running
    st.markdown(
        f'<div style="text-align:center;margin-bottom:14px;">'
        f'<span style="font-size:14px;font-weight:700;">'
        f'{"يعمل 24/7 🟢" if running else "متوقف 🔴"}</span>'
        f'<span class="pulse{"" if running else " off"}"></span></div>',
        unsafe_allow_html=True)

    budget = st.number_input("💰 الميزانية (USDT)",
        min_value=5.0, max_value=10000.0,
        value=float(st.session_state.budget), step=5.0)
    st.session_state.budget = budget

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sb">', unsafe_allow_html=True)
        if st.button("▶ تشغيل", use_container_width=True):
            start_bot(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="xb">', unsafe_allow_html=True)
        if st.button("■ إيقاف", use_container_width=True):
            stop_bot(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # زر السحب
    if st.session_state.withdraw_ready:
        st.markdown('<div class="wb">', unsafe_allow_html=True)
        if st.button("💸 سحب الأرباح الآن", use_container_width=True):
            do_withdraw(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    bal  = st.session_state.balance
    init = st.session_state.initial_capital
    wdrn = st.session_state.withdrawn
    pct  = (bal - init) / init * 100 if init > 0 else 0
    col  = "#00ff9d" if bal >= init else "#ff3860"

    st.markdown(
        f'<div class="ml">الرصيد المتداول</div>'
        f'<div style="font-size:22px;font-weight:900;color:{col};'
        f'font-family:JetBrains Mono,monospace;">${bal:.4f}</div>'
        f'<div style="font-size:11px;color:{col};margin-bottom:6px;">{pct:+.2f}%</div>',
        unsafe_allow_html=True)

    # شريط تقدم المضاعفة
    prog = min((bal + wdrn - init) / init * 100, 100) if init > 0 else 0
    st.markdown(
        f'<div class="ml">تقدم المضاعفة ({prog:.1f}%)</div>'
        f'<div class="prog-wrap"><div class="prog-bar" style="width:{prog}%"></div></div>',
        unsafe_allow_html=True)

    if wdrn > 0:
        st.markdown(
            f'<div style="margin-top:8px;" class="ml">إجمالي المسحوب</div>'
            f'<div style="font-size:16px;font-weight:700;color:#ffd600;'
            f'font-family:JetBrains Mono,monospace;">${wdrn:.4f}</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    n_sym = len(st.session_state.all_symbols)
    st.markdown(f"""
    <div style="font-size:11px;color:#3a6a8a;line-height:2.2;">
    🔍 عملات مفحوصة: <b style="color:#00c8ff;">{n_sym}</b><br>
    📊 MFI + RSI + EMA + ATR<br>
    🌙 وضع القمر: AI > 82%<br>
    🎯 TP عادي: 1.8% | قمر: ∞<br>
    🛡 SL: 1.2% | Trailing: 0.7%<br>
    💸 عمولة: 0.2% × 2<br>
    🔄 Compound: تلقائي<br>
    ⏰ يعمل: 24/7 بلا توقف
    </div>""", unsafe_allow_html=True)

    if st.button("🔄 تحديث", use_container_width=True):
        st.rerun()

# ════════════════════════════════════════════════════════════
#  لوحة التحكم
# ════════════════════════════════════════════════════════════
st.markdown(
    '<h1 style="font-family:Cairo,sans-serif;font-size:24px;font-weight:900;'
    'color:#00c8ff;margin:0 0 2px;">🚀 مركز التحكم — بوت المضاربة v3.0</h1>'
    '<div style="font-size:11px;color:#3a6a8a;margin-bottom:14px;">'
    '⚡ فحص شامل لجميع العملات الرخيصة | Compound تلقائي | وضع القمر 🌙 | 24/7</div>',
    unsafe_allow_html=True)

# تنبيه سحب
if st.session_state.withdraw_ready:
    bal  = st.session_state.balance
    init = st.session_state.initial_capital
    prof = bal - init
    st.markdown(
        f'<div class="withdraw-banner">'
        f'<div style="font-size:16px;font-weight:900;color:#ffd600;">🎉 رأس المال تضاعف!</div>'
        f'<div style="font-size:13px;margin-top:6px;">البداية: ${init:.2f} | الرصيد: ${bal:.2f} | الربح: ${prof:.2f}</div>'
        f'<div style="font-size:12px;color:#aa8800;margin-top:4px;">اضغط زر "سحب الأرباح" في الشريط الجانبي</div>'
        f'</div>',
        unsafe_allow_html=True)

# تنبيهات لحظية
for a in list(st.session_state.alerts)[:2]:
    st.markdown(
        f'<div style="background:#002a1a;border:1px solid #00ff9d;border-radius:8px;'
        f'padding:10px 16px;margin-bottom:6px;font-size:13px;color:#00ff9d;">'
        f'🔔 [{a["time"]}] {a["msg"]}</div>',
        unsafe_allow_html=True)

# ── المقاييس العلوية ──
s     = st.session_state.stats
total = s["win"] + s["loss"]
wr    = s["win"]/total*100 if total>0 else 0

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
def mc(col, label, val, sub="", css=""):
    with col:
        st.markdown(
            f'<div class="mc {css}">'
            f'<div class="ml">{label}</div>'
            f'<div class="mv">{val}</div>'
            f'<div class="ms">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

pnl_d = s["pnl_today"]
pnl_t = s["pnl_total"]
mc(c1,"ربح اليوم",f'{"+" if pnl_d>=0 else ""}{pnl_d:.3f}$',"صافٍ","g" if pnl_d>=0 else "r")
mc(c2,"الربح الكلي",f'{"+" if pnl_t>=0 else ""}{pnl_t:.3f}$',"منذ البداية","g" if pnl_t>=0 else "r")
mc(c3,"ربح/خسارة",f'{s["win"]}/{s["loss"]}',f'{total} صفقة',"y")
mc(c4,"نسبة النجاح",f'{wr:.1f}%',"AI فعّال ✓" if st.session_state.ai_trained else "يتعلم...","g" if wr>=50 else "r")
mc(c5,"مفتوحة",f'{len(st.session_state.open_trades)}/{MAX_OPEN}',"صفقات","o")
mc(c6,"🌙 وضع القمر",f'{s["moon_shots"]}','صفقات كبيرة',"p")
mc(c7,"عملات مفحوصة",f'{s["total_scanned"]:,}',"إجمالي","")

# ── الفحص المباشر ──
st.markdown('<div class="sh">🔍 الفحص المباشر — ماذا يحدث الآن؟</div>', unsafe_allow_html=True)

cur = st.session_state.current_scan
cyc = st.session_state.scan_cycle
nsym = len(st.session_state.all_symbols)

col_a, col_b = st.columns([3,1])
with col_a:
    if cur:
        st.markdown(
            f'<div class="sc a">'
            f'<div><span class="sym">⟳ {cur}</span>'
            f'<div class="rsn">جارٍ التحليل الآن...</div></div>'
            f'<span class="badge b-scan">يفحص</span></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="sc" style="justify-content:center;color:#3a6a8a;">'
            f'{"في انتظار الدورة التالية..." if st.session_state.running else "شغّل البوت للبدء"}'
            f'</div>', unsafe_allow_html=True)

with col_b:
    st.markdown(
        f'<div class="mc">'
        f'<div class="ml">دورة الفحص</div>'
        f'<div class="mv" style="font-size:18px;">#{cyc}</div>'
        f'<div class="ms">{nsym} عملة/دورة</div>'
        f'</div>', unsafe_allow_html=True)

# آخر نتائج الفحص
scan_hist = list(st.session_state.scan_results)[:12]
if scan_hist:
    for r in scan_hist:
        is_moon_r = r["status"] == "moon"
        css_card  = "moon" if is_moon_r else ("p" if r["status"]=="pass" else "f")
        b_cls     = "b-moon" if is_moon_r else ("b-pass" if r["status"]=="pass" else "b-fail")
        b_txt     = "🌙 قمر" if is_moon_r else ("✓ دخلنا" if r["status"]=="pass" else "✗ رُفض")
        bar_w     = int(r["ai_prob"] * 55)
        st.markdown(
            f'<div class="sc {css_card}">'
            f'<div style="flex:1;">'
            f'<span class="sym">{r["symbol"]}</span>'
            f'{"<span style=\'font-size:10px;color:#c084fc;margin-right:6px;\'>🌙</span>" if is_moon_r else ""}'
            f'<div class="rsn">{r["reason"]}</div>'
            f'<div style="display:flex;align-items:center;gap:6px;margin-top:3px;">'
            f'<div style="width:{bar_w}px;height:4px;border-radius:2px;'
            f'background:linear-gradient(90deg,#00c8ff,#00ff9d);"></div>'
            f'<span style="font-size:10px;color:#4a6a8a;">{r["ai_prob"]*100:.0f}%</span>'
            f'</div></div>'
            f'<div style="text-align:left;">'
            f'<span class="badge {b_cls}">{b_txt}</span>'
            f'<div style="font-size:10px;color:#3a6a8a;margin-top:3px;">{r["time"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True)

# ── الصفقات المفتوحة ──
st.markdown('<div class="sh">🔴 الصفقات المفتوحة</div>', unsafe_allow_html=True)
if st.session_state.open_trades:
    rows = []
    for sym, t in st.session_state.open_trades.items():
        tk  = fetch_ticker(sym)
        cur = tk["last"] if tk else t["entry_price"]
        net = ((cur-t["entry_price"])*t["qty"] - t["fee_in"] - t["qty"]*cur*COMMISSION)
        pnl = net/t["alloc"]*100
        rows.append({
            "العملة":       sym,
            "النوع":        "🌙 قمر" if t["is_moon"] else "⚡ عادي",
            "الدخول $":    f'{t["entry_price"]:.8f}',
            "الحالي $":    f'{cur:.8f}',
            "أعلى ربح":   f'{t["peak_gain"]:+.2f}%',
            "وقف متحرك":  f'{t["trail_sl"]:.8f}',
            "PnL صافٍ":   f'{pnl:+.3f}%',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div style="text-align:center;padding:16px;color:#3a6a8a;">⊘ لا صفقات مفتوحة</div>',
        unsafe_allow_html=True)

# ── أفضل الفرص الآن ──
st.markdown('<div class="sh">📡 أعلى إشارات الذكاء الاصطناعي</div>', unsafe_allow_html=True)
if st.session_state.scan_list:
    top = st.session_state.scan_list[:10]
    rows = []
    for s2 in top:
        rows.append({
            "العملة":     s2["symbol"],
            "السعر $":   f'{s2["price"]:.8f}',
            "MFI":        s2["mfi"],
            "حجم ×":     f'×{s2["vol_ratio"]}',
            "زخم":        f'{s2["momentum"]:+.3f}%',
            "AI %":       f'{s2["ai_prob"]}%',
            "الإشارة":    s2["signal"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── سجل الصفقات ──
st.markdown('<div class="sh">📋 سجل الصفقات</div>', unsafe_allow_html=True)
if st.session_state.trades:
    rows = []
    for t in reversed(st.session_state.trades[-60:]):
        pnl = t.get("pnl_pct",0) or 0
        net = t.get("pnl_net",0) or 0
        rows.append({
            "العملة":    t["symbol"],
            "النوع":     "🌙" if t.get("is_moon") else "⚡",
            "الدخول":   t["entry_time"],
            "دخول $":   f'{t["entry_price"]:.8f}',
            "خروج $":   f'{t["exit_price"]:.8f}' if t["exit_price"] else "—",
            "PnL %":    f'{pnl:+.3f}%',
            "صافٍ $":   f'{net:+.6f}',
            "الحالة":    "🟢 ربح" if t["status"]=="WIN" else "🔴 خسارة",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    a1,a2,a3,a4 = st.columns(4)
    s2 = st.session_state.stats
    for col,label,val,css in [
        (a1,"أفضل صفقة",f'{s2["best_trade"]:+.2f}%',"g"),
        (a2,"أسوأ صفقة",f'{s2["worst_trade"]:+.2f}%',"r"),
        (a3,"🌙 صفقات القمر",str(s2["moon_shots"]),"p"),
        (a4,"إجمالي العمولات",f'${s2["total_fees"]:.4f}',"r"),
    ]:
        with col:
            st.markdown(
                f'<div class="mc {css}"><div class="ml">{label}</div>'
                f'<div class="mv" style="font-size:18px;">{val}</div></div>',
                unsafe_allow_html=True)

# ── سجل Compound ──
if st.session_state.compound_log:
    st.markdown('<div class="sh">💰 سجل المضاعفة</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state.compound_log),
                 use_container_width=True, hide_index=True)

# ── سجل الأحداث ──
st.markdown('<div class="sh">📟 سجل الأحداث</div>', unsafe_allow_html=True)
filt = st.selectbox("عرض:",
    ["الكل","الصفقات","نتائج الفحص","التحذيرات","القمر 🌙"],
    label_visibility="collapsed")
fmap = {
    "الكل":         None,
    "الصفقات":      ["buy","sell","moon","withdraw","compound"],
    "نتائج الفحص":  ["pass","fail","scan"],
    "التحذيرات":    ["warn"],
    "القمر 🌙":     ["moon"],
}
chosen = fmap[filt]
logs_show = [l for l in st.session_state.logs if chosen is None or l["kind"] in chosen]
if logs_show:
    html = '<div class="log-box">'
    for e in logs_show[:60]:
        html += f'<div class="log-{e["kind"]}">[{e["ts"]}] {e["msg"]}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ── حالة AI ──
st.markdown('<div class="sh">🧠 الذكاء الاصطناعي</div>', unsafe_allow_html=True)
a1,a2,a3,a4 = st.columns(4)
trained = st.session_state.ai_trained
n_hist  = len(st.session_state.ai_history)
prog_ai = min(n_hist/20*100,100)
for col,label,val,sub,css in [
    (a1,"الحالة","✓ مُدرَّب" if trained else "⟳ يتعلم","Gradient Boosting","g" if trained else "y"),
    (a2,"عينات التدريب",str(n_hist),f"تقدم: {prog_ai:.0f}%",""),
    (a3,"المؤشرات","7","MFI،RSI،EMA،ATR،حجم،زخم،اتجاه",""),
    (a4,"عتبة القمر 🌙","82%","AI لصفقات بلا سقف ربح","p"),
]:
    with col:
        st.markdown(
            f'<div class="mc {css}"><div class="ml">{label}</div>'
            f'<div class="mv" style="font-size:16px;">{val}</div>'
            f'<div class="ms">{sub}</div></div>',
            unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:10px;color:#1a3050;font-family:Cairo,sans-serif;">'
    '⚡ بوت المضاربة v3.0 — وضع التداول الوهمي — ليس نصيحة مالية ⚡</div>',
    unsafe_allow_html=True)

# تحديث تلقائي كل 8 ثوانٍ
if st.session_state.running:
    time.sleep(8)
    st.rerun()
