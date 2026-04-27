import streamlit as st
import ccxt, pandas as pd, sqlite3, time, threading, json, math

# ========= CONFIG =========
ADMIN_KEY = "123456"
TRAILING_STOP = 0.02
LOOP_INTERVAL = 10

# ========= DB =========
conn = sqlite3.connect("bot.db", check_same_thread=False)
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS balance (id INTEGER PRIMARY KEY, value REAL)")
c.execute("CREATE TABLE IF NOT EXISTS positions (symbol TEXT, entry REAL, max_price REAL)")
c.execute("CREATE TABLE IF NOT EXISTS trades (symbol TEXT, entry REAL, exit REAL, profit REAL, time TEXT)")
c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
conn.commit()

def get_balance():
    r = c.execute("SELECT value FROM balance WHERE id=1").fetchone()
    if r: return r[0]
    c.execute("INSERT INTO balance VALUES (1,100)")
    conn.commit()
    return 100

def set_balance(v):
    c.execute("UPDATE balance SET value=? WHERE id=1",(v,))
    conn.commit()

def get_amount():
    r = c.execute("SELECT value FROM settings WHERE key='amount'").fetchone()
    return float(r[0]) if r else None

def set_amount(v):
    c.execute("INSERT OR IGNORE INTO settings VALUES ('amount',?)",(v,))
    conn.commit()

def get_positions():
    rows = c.execute("SELECT * FROM positions").fetchall()
    return {r[0]: {"entry": r[1], "max": r[2]} for r in rows}

def save_position(s,p):
    c.execute("INSERT INTO positions VALUES (?,?,?)",(s,p,p))
    conn.commit()

def update_max(s,p):
    c.execute("UPDATE positions SET max_price=? WHERE symbol=?",(p,s))
    conn.commit()

def remove_position(s):
    c.execute("DELETE FROM positions WHERE symbol=?",(s,))
    conn.commit()

def save_trade(s,e,x,profit):
    c.execute("INSERT INTO trades VALUES (?,?,?,?,datetime('now'))",(s,e,x,profit))
    conn.commit()

def get_trades():
    return c.execute("SELECT * FROM trades ORDER BY time DESC").fetchall()

# ========= AI MODEL =========
MODEL_FILE = "ai_model.json"

def load_model():
    try:
        return json.load(open(MODEL_FILE))
    except:
        return {"w":[0.2]*6,"b":0}

def save_model(m):
    json.dump(m,open(MODEL_FILE,"w"))

def sigmoid(x):
    return 1/(1+math.exp(-x))

def predict(m,features):
    z = m["b"]
    for i,f in enumerate(features):
        z += m["w"][i]*f
    return sigmoid(z)

def train(m,features,label,lr=0.05):
    p = predict(m,features)
    error = label - p
    for i in range(len(m["w"])):
        m["w"][i] += lr * error * features[i]
    m["b"] += lr * error
    save_model(m)

# ========= MARKET =========
ex = ccxt.mexc({"enableRateLimit": True})

def get_symbols():
    tickers = ex.fetch_tickers()
    pairs = []
    for s,t in tickers.items():
        if "/USDT" in s and t.get("last") and t["last"] <= 0.001:
            if t.get("quoteVolume",0) > 20000:
                pairs.append((s, t["quoteVolume"]))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs[:12]]

def get_df(s):
    bars = ex.fetch_ohlcv(s,'1m',limit=50)
    df = pd.DataFrame(bars,columns=['ts','o','h','l','c','v'])
    df['ema8']=df['c'].ewm(span=8).mean()
    df['ema21']=df['c'].ewm(span=21).mean()
    df['rsi']=100-(100/(1+df['c'].pct_change().rolling(14).mean()))
    df['vol_avg']=df['v'].rolling(10).mean()
    return df

def score(df):
    last=df.iloc[-1]; prev=df.iloc[-2]
    s=0
    if last['ema8']>last['ema21']: s+=25
    if last['ema8']>last['ema21'] and prev['ema8']<=prev['ema21']: s+=25
    if 25<last['rsi']<40: s+=15
    if last['c']>prev['c']: s+=15
    if last['v']>last['vol_avg']: s+=10
    if abs(last['c']-prev['c'])/prev['c']<0.02: s+=10
    return s

def extract_features(df):
    last=df.iloc[-1]; prev=df.iloc[-2]
    return [
        (last['ema8']-last['ema21'])/last['c'],
        last['rsi']/100,
        (last['c']-prev['c'])/prev['c'],
        last['v']/(df['v'].rolling(10).mean().iloc[-1]+1e-9),
        abs(last['c']-prev['c'])/prev['c'],
        1 if last['ema8']>last['ema21'] else 0
    ]

# ========= BOT =========
def bot():
    print("BOT STARTED")
    while True:
        try:
            amt = get_amount()
            if not amt:
                time.sleep(5)
                continue

            balance = get_balance()
            pos = get_positions()
            symbols = get_symbols()
            model = load_model()

            best = None

            for s in symbols:
                df = get_df(s)
                sc = score(df)
                feats = extract_features(df)
                ai = predict(model, feats)
                combined = sc * (0.7 + 0.6 * ai)

                price = df.iloc[-1]['c']

                # خروج
                if s in pos:
                    entry = pos[s]['entry']
                    maxp = pos[s]['max']

                    if price > maxp:
                        update_max(s, price)
                        continue

                    drop = (maxp - price) / maxp

                    if drop >= TRAILING_STOP or ai < 0.35:
                        profit = (price - entry) / entry
                        balance += amt * (1 + profit)
                        set_balance(balance)
                        save_trade(s, entry, price, profit)
                        remove_position(s)
                        train(model, feats, 1 if profit>0 else 0)
                        print("SELL", s, profit)

                # اختيار أفضل صفقة
                elif sc >= 70:
                    if not best or combined > best["score"]:
                        best = {"s": s, "p": price, "f": feats, "score": combined}

            # دخول صفقة واحدة فقط
            if len(pos)==0 and best and balance>=amt:
                set_balance(balance-amt)
                save_position(best["s"], best["p"])
                st.session_state["last_feats"] = best["f"]
                print("BUY", best["s"])

        except Exception as e:
            print("ERR", e)

        time.sleep(LOOP_INTERVAL)

# تشغيل مرة واحدة
if "run" not in st.session_state:
    threading.Thread(target=bot, daemon=True).start()
    st.session_state.run = True

# ========= UI =========
st.title("📊 AI Trading Bot")

amt = st.number_input("مبلغ التداول", value=10.0)
key = st.text_input("Admin Key")

if st.button("تثبيت"):
    if key != ADMIN_KEY:
        st.error("مفتاح خاطئ")
    elif get_amount():
        st.warning("تم التثبيت مسبقاً")
    else:
        set_amount(amt)
        st.success("تم التثبيت")

balance = get_balance()
trades = get_trades()

st.metric("الرصيد", balance)

wins = sum(1 for t in trades if t[3] > 0)
loss = sum(1 for t in trades if t[3] <= 0)
total = len(trades)

st.write("عدد الصفقات:", total)
st.write("الرابحة:", wins)
st.write("الخاسرة:", loss)

if total:
    st.write("نسبة النجاح:", round(wins/total*100,2), "%")

st.subheader("الصفقات")
for t in trades:
    st.write(t)
