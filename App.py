import streamlit as st
import ccxt, pandas as pd, sqlite3, threading, time, math

# ================= CONFIG =================
ADMIN_KEY = "123456"
LOOP_INTERVAL = 10
TRAILING_STOP = 0.02

# ================= STATE =================
if "running" not in st.session_state:
    st.session_state.running = False

if "state" not in st.session_state:
    st.session_state.state = {
        "checked": 0,
        "current": None,
        "buy": None,
        "sell": None
    }

# ================= DB SAFE =================
def conn():
    return sqlite3.connect("bot.db", check_same_thread=False)

def init_db():
    c = conn().cursor()
    c.execute("CREATE TABLE IF NOT EXISTS balance (id INTEGER PRIMARY KEY, value REAL)")
    c.execute("CREATE TABLE IF NOT EXISTS positions (symbol TEXT, entry REAL, max_price REAL)")
    c.execute("CREATE TABLE IF NOT EXISTS trades (symbol TEXT, entry REAL, exit REAL, profit REAL, time TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    conn().commit()

init_db()

# ================= DB OPS =================
def get_balance():
    c = conn().cursor()
    r = c.execute("SELECT value FROM balance WHERE id=1").fetchone()
    if r:
        return r[0]
    c.execute("INSERT INTO balance VALUES (1,100)")
    conn().commit()
    return 100

def set_balance(v):
    c = conn().cursor()
    c.execute("UPDATE balance SET value=? WHERE id=1",(v,))
    conn().commit()

def get_amount():
    c = conn().cursor()
    r = c.execute("SELECT value FROM settings WHERE key='amount'").fetchone()
    return float(r[0]) if r else None

def set_amount(v):
    c = conn().cursor()
    c.execute("INSERT OR IGNORE INTO settings VALUES ('amount',?)",(v,))
    conn().commit()

def get_positions():
    c = conn().cursor()
    rows = c.execute("SELECT * FROM positions").fetchall()
    return {r[0]:{"entry":r[1],"max":r[2]} for r in rows}

def save_pos(s,p):
    c = conn().cursor()
    c.execute("INSERT INTO positions VALUES (?,?,?)",(s,p,p))
    conn().commit()

def update_max(s,p):
    c = conn().cursor()
    c.execute("UPDATE positions SET max_price=? WHERE symbol=?",(p,s))
    conn().commit()

def remove_pos(s):
    c = conn().cursor()
    c.execute("DELETE FROM positions WHERE symbol=?",(s,))
    conn().commit()

def save_trade(s,e,x,p):
    c = conn().cursor()
    c.execute("INSERT INTO trades VALUES (?,?,?,?,datetime('now'))",(s,e,x,p))
    conn().commit()

def get_trades():
    c = conn().cursor()
    return c.execute("SELECT * FROM trades ORDER BY time DESC").fetchall()

# ================= MARKET =================
ex = ccxt.mexc({"enableRateLimit": True})

def get_symbols():
    t = ex.fetch_tickers()
    out = []
    for s,v in t.items():
        if "/USDT" in s and v.get("last") and v["last"]<=0.001:
            if v.get("quoteVolume",0)>20000:
                out.append(s)
    return out[:12]

def get_df(s):
    b = ex.fetch_ohlcv(s,'1m',limit=50)
    df = pd.DataFrame(b,columns=['t','o','h','l','c','v'])
    df['ema8']=df['c'].ewm(span=8).mean()
    df['ema21']=df['c'].ewm(span=21).mean()
    df['rsi']=100-(100/(1+df['c'].pct_change().rolling(14).mean()))
    df['vol_avg']=df['v'].rolling(10).mean()
    return df

def score(df):
    l=df.iloc[-1]; p=df.iloc[-2]
    s=0
    if l['ema8']>l['ema21']: s+=25
    if l['ema8']>l['ema21'] and p['ema8']<=p['ema21']: s+=25
    if 25<l['rsi']<40: s+=15
    if l['c']>p['c']: s+=15
    if l['v']>l['vol_avg']: s+=10
    if abs(l['c']-p['c'])/p['c']<0.02: s+=10
    return s

# ================= BOT =================
def bot():
    while st.session_state.running:
        try:
            amt = get_amount()
            if not amt:
                time.sleep(3)
                continue

            bal = get_balance()
            pos = get_positions()
            syms = get_symbols()

            for s in syms:
                st.session_state.state["checked"] += 1
                st.session_state.state["current"] = s

                df = get_df(s)
                sc = score(df)
                price = df.iloc[-1]['c']

                # ===== SELL =====
                if s in pos:
                    entry = pos[s]['entry']
                    maxp = pos[s]['max']

                    if price > maxp:
                        update_max(s,price)
                        continue

                    drop = (maxp-price)/maxp

                    if drop >= TRAILING_STOP:
                        profit = (price-entry)/entry
                        bal += amt*(1+profit)
                        set_balance(bal)
                        save_trade(s,entry,price,profit)
                        remove_pos(s)

                        st.session_state.state["sell"] = price

                # ===== BUY =====
                elif len(pos)==0 and sc>=75 and bal>=amt:
                    set_balance(bal-amt)
                    save_pos(s,price)

                    st.session_state.state["buy"] = price

        except Exception as e:
            print("ERR",e)

        time.sleep(LOOP_INTERVAL)

# ================= UI =================
st.title("📊 PRO TRADING BOT")

col1,col2 = st.columns(2)

if col1.button("▶ تشغيل"):
    if not st.session_state.running:
        st.session_state.running = True
        threading.Thread(target=bot,daemon=True).start()

if col2.button("⛔ إيقاف"):
    st.session_state.running = False

st.divider()

st.metric("عدد العملات المفحوصة", st.session_state.state["checked"])
st.write("العملة الحالية:", st.session_state.state["current"])
st.write("سعر الشراء:", st.session_state.state["buy"])
st.write("سعر البيع:", st.session_state.state["sell"])

st.metric("الرصيد", get_balance())

# ===== إعداد المبلغ =====
amt = st.number_input("مبلغ التداول",value=10.0)
key = st.text_input("Admin Key")

if st.button("تثبيت المبلغ"):
    if key != ADMIN_KEY:
        st.error("مفتاح خاطئ")
    elif get_amount():
        st.warning("تم التثبيت مسبقاً")
    else:
        set_amount(amt)
        st.success("تم التثبيت")

# ===== الصفقات =====
st.subheader("الصفقات")

for t in get_trades():
    st.write(f"{t[0]} | دخول {t[1]} | خروج {t[2]} | ربح {round(t[3]*100,2)}%")
