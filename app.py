import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from openai import OpenAI
import datetime

st.set_page_config(
    page_title="Engine AI Diagnostic",
    page_icon="🔧",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #0d1117;
    background-image:
        radial-gradient(circle at 90% -5%, #1a2744 0%, transparent 45%),
        radial-gradient(circle at -5% 95%, #0f2a1a 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, #0d1117 0%, #0d1117 100%);
}

section[data-testid="stSidebar"] {
    background: #010409 !important;
    border-right: 1px solid #21262d !important;
}

section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: #1a2744;
    top: -30px; right: -30px;
    pointer-events: none;
}

.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 1.2rem 0; margin-bottom: 1.2rem;
    border-bottom: 1px solid #21262d;
}
.nav-brand { display: flex; align-items: center; gap: 12px; }
.nav-icon {
    width: 36px; height: 36px; background: #238636;
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 18px;
}
.nav-title { color: #e6edf3; font-size: 17px; font-weight: 600; }
.nav-badge {
    background: #1f6feb22; color: #58a6ff;
    border: 1px solid #1f6feb55; font-size: 11px;
    padding: 3px 10px; border-radius: 20px; font-weight: 500;
}

.status-healthy {
    background: #0f2a1a; color: #4ade80;
    border: 1px solid #166534; padding: 1rem 1.4rem;
    border-radius: 10px; font-size: 1.3rem; font-weight: 700;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 10px;
}
.status-atrisk {
    background: #2a1f0a; color: #fbbf24;
    border: 1px solid #92400e; padding: 1rem 1.4rem;
    border-radius: 10px; font-size: 1.3rem; font-weight: 700;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 10px;
}
.status-critical {
    background: #2a0a0a; color: #f87171;
    border: 1px solid #991b1b; padding: 1rem 1.4rem;
    border-radius: 10px; font-size: 1.3rem; font-weight: 700;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 10px;
}

.metric-grid { display: flex; gap: 10px; margin-bottom: 1rem; }
.metric-card {
    flex: 1; background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 12px 14px;
}
.metric-label {
    font-size: 10px; color: #484f58;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;
}
.metric-value { font-size: 20px; font-weight: 500; color: #e6edf3; }

.section-title {
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #484f58;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px; margin: 1rem 0 0.8rem;
}

.obd-found {
    background: #0f1f2a; border: 1px solid #1d4e6e;
    border-radius: 8px; padding: 0.8rem 1rem;
    color: #58a6ff; font-size: 0.88rem; margin-top: 0.5rem;
}

.ai-box {
    background: #161b22; border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0; padding: 1rem 1.2rem;
    color: #c9d1d9; font-size: 0.9rem; line-height: 1.75;
    margin-top: 0.5rem;
}

.hist-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 7px 0;
    border-bottom: 1px solid #21262d; font-size: 11px;
}
.hist-row:last-child { border-bottom: none; }
.hist-time { color: #484f58; width: 70px; }
.hist-status { font-weight: 600; width: 100px; }
.hist-readings { color: #8b949e; flex: 1; text-align: center; }
.hist-cost { color: #8b949e; text-align: right; }

.geo-accent {
    position: fixed; width: 60px; height: 60px;
    background: #1d3a5c; transform: rotate(45deg);
    border-radius: 6px; opacity: 0.4;
    top: 220px; right: 120px; pointer-events: none;
}

div[data-testid="stButton"] > button {
    background: #238636 !important; color: white !important;
    border: none !important; font-weight: 600 !important;
    font-size: 14px !important; padding: 0.6rem 1.5rem !important;
    border-radius: 6px !important; width: 100% !important;
    transition: background 0.2s !important;
}
div[data-testid="stButton"] > button:hover { background: #2ea043 !important; }

.stSlider > div > div > div { background: #238636 !important; }
label { color: #8b949e !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── API KEY ────────────────────────────────────────────────────────
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    api_key = "YOUR_OPENAI_API_KEY_HERE"

client = OpenAI(api_key=api_key)

# ── SESSION STATE ──────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── OBD LIBRARY ───────────────────────────────────────────────────
obd_library = {
    'P0217': ('Engine Over Temperature',            'Engine Cooling System'),
    'P0128': ('Coolant Temp Below Thermostat',       'Engine Cooling System'),
    'P0520': ('Oil Pressure Sensor Circuit',         'Lubrication System'),
    'P0521': ('Oil Pressure Sensor Range/Perf',      'Lubrication System'),
    'P0087': ('Fuel Rail Pressure Too Low',          'Fuel System'),
    'P0088': ('Fuel Rail Pressure Too High',         'Fuel System'),
    'P0171': ('System Too Lean Bank 1',              'Fuel System'),
    'P0172': ('System Too Rich Bank 1',              'Fuel System'),
    'P0301': ('Cylinder 1 Misfire Detected',         'Ignition System'),
    'P0302': ('Cylinder 2 Misfire Detected',         'Ignition System'),
    'P0303': ('Cylinder 3 Misfire Detected',         'Ignition System'),
    'P0304': ('Cylinder 4 Misfire Detected',         'Ignition System'),
    'P0420': ('Catalyst Efficiency Below Threshold', 'Emission System'),
    'P0113': ('Intake Air Temp Sensor High Input',   'Air Intake System'),
    'P0136': ('O2 Sensor Circuit Fault',             'Emission System'),
}

def get_code_description(code):
    code = code.upper().strip()
    if code in obd_library:
        return obd_library[code]
    return ('Code not found', None)

# ── TRAIN MODEL ────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    data = [
        [2500,90,35,55,0],[3000,95,40,58,0],[1800,88,32,50,0],
        [2000,92,38,54,0],[3500,99,36,57,0],[2800,93,41,56,0],
        [1500,85,30,52,0],[2700,91,37,53,0],
        [2200,107,22,42,1],[4000,110,18,38,1],[4500,108,20,40,1],
        [3800,112,17,36,1],[5000,106,23,39,1],[3200,109,19,37,1],
        [4200,111,21,41,1],[3600,107,24,43,1],
        [600,120,10,28,2],[550,125,8,25,2],[580,128,7,22,2],
        [620,122,9,26,2],[500,130,6,20,2],[520,132,5,18,2],
        [480,135,4,15,2],[490,127,6,21,2],
    ]
    df = pd.DataFrame(data, columns=['rpm','coolant','oil','fuel','condition'])
    m  = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(df[['rpm','coolant','oil','fuel']], df['condition'])
    return m

model = train_model()

# ── SIDEBAR ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid #21262d;">
        <div style="width:34px;height:34px;background:#238636;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:17px;">🔧</div>
        <span style="color:#e6edf3;font-size:15px;font-weight:600;">Engine AI</span>
        <span style="background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb55;font-size:10px;padding:2px 8px;border-radius:20px;">Live</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sensor Readings</div>', unsafe_allow_html=True)
    rpm   = st.slider("Engine RPM",          500,  7000, 2500, 50)
    cool  = st.slider("Coolant Temp (°C)",   50,   135,  90,   1)
    oil   = st.slider("Oil Pressure (PSI)",  0,    100,  35,   1)
    fuel  = st.slider("Fuel Pressure (PSI)", 20,   100,  55,   1)

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Optional</div>', unsafe_allow_html=True)
    obd_code = st.text_input("OBD Code", placeholder="e.g. P0301")
    question = st.text_area("Ask the AI", placeholder="e.g. Why is my engine overheating?", height=90)
    run = st.button("Run Diagnosis")

# ── MAIN ───────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-brand">
        <div class="nav-icon">🔧</div>
        <span class="nav-title">Engine AI Diagnostic</span>
    </div>
    <span class="nav-badge">Powered by AI</span>
</div>
""", unsafe_allow_html=True)

col_results, col_charts = st.columns([1, 1.3], gap="large")

with col_results:
    if run:
        reading = pd.DataFrame([[rpm,cool,oil,fuel]], columns=['rpm','coolant','oil','fuel'])
        pred    = model.predict(reading)[0]
        proba   = model.predict_proba(reading)[0]

        status_map = {0:'🟢  HEALTHY', 1:'🟡  AT RISK', 2:'🔴  CRITICAL'}
        class_map  = {0:'status-healthy', 1:'status-atrisk', 2:'status-critical'}
        cost_map   = {0:'$0 – $150', 1:'$150 – $900', 2:'$900 – $5,000+'}
        action_map = {
            0: 'No immediate action needed. Keep up regular maintenance.',
            1: 'Schedule an inspection within 1–2 weeks.',
            2: 'Stop driving immediately. Risk of serious engine damage.'
        }

        st.markdown(f'<div class="{class_map[pred]}">{status_map[pred]}</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Est. Repair Cost</div>
                <div class="metric-value">{cost_map[pred]}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{max(proba):.0%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info(action_map[pred])

        st.markdown('<div class="section-title">Prediction Breakdown</div>', unsafe_allow_html=True)
        labels = ['Healthy','At Risk','Critical']
        for lbl, p in zip(labels, proba):
            st.progress(float(p), text=f"{lbl}  {p:.0%}")

        if obd_code:
            st.markdown('<div class="section-title">OBD Code Lookup</div>', unsafe_allow_html=True)
            desc, system = get_code_description(obd_code)
            if desc != 'Code not found':
                st.markdown(f'<div class="obd-found"><strong>{obd_code.upper()}</strong> — {desc}<br><span style="opacity:0.6;font-size:11px;">System: {system}</span></div>',
                            unsafe_allow_html=True)
            else:
                st.warning(f"{obd_code.upper()} not found in library.")

        if question:
            st.markdown('<div class="section-title">AI Diagnosis</div>', unsafe_allow_html=True)
            with st.spinner("Thinking..."):
                obd_info = obd_library.get(obd_code.upper().strip(),(None,None)) if obd_code else (None,None)
                prompt = f"""Engine diagnostics — RPM={rpm}, Coolant={cool}C, Oil={oil}PSI, Fuel={fuel}PSI.
Status: {status_map[pred]}. OBD: {obd_code or 'None'}. Description: {obd_info[0]}. System: {obd_info[1]}.
Question: {question}"""
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":"You are an expert automotive diagnostic technician. Give clear, practical, plain-English advice in 3-5 sentences."},
                        {"role":"user","content":prompt}
                    ]
                )
                answer = resp.choices[0].message.content
            st.markdown(f'<div class="ai-box">{answer}</div>', unsafe_allow_html=True)

        st.session_state.history.append({
            'time':    datetime.datetime.now().strftime('%H:%M:%S'),
            'status':  status_map[pred],
            'rpm':     rpm, 'coolant': cool,
            'oil':     oil, 'fuel':    fuel,
            'cost':    cost_map[pred],
            'obd':     obd_code.upper() if obd_code else '—'
        })

    else:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:2rem;text-align:center;margin-top:1rem;">
            <div style="font-size:2rem;margin-bottom:0.8rem;">🔧</div>
            <div style="color:#e6edf3;font-size:15px;font-weight:500;margin-bottom:0.5rem;">Ready to Diagnose</div>
            <div style="color:#484f58;font-size:13px;">Set your sensor readings on the left<br>and click Run Diagnosis</div>
        </div>
        """, unsafe_allow_html=True)

with col_charts:
    chart_data = [
        [2500,90,35,55,0],[3000,95,40,58,0],[1800,88,32,50,0],[2000,92,38,54,0],
        [2200,107,22,42,1],[4000,110,18,38,1],[4500,108,20,40,1],[3800,112,17,36,1],
        [600,120,10,28,2],[550,125,8,25,2],[580,128,7,22,2],[500,130,6,20,2],
    ]
    df_chart = pd.DataFrame(chart_data, columns=['RPM','Coolant','Oil','Fuel','Condition'])
    df_chart['Status'] = df_chart['Condition'].map({0:'Healthy',1:'At Risk',2:'Critical'})

    if run:
        your = pd.DataFrame([[rpm,cool,oil,fuel,-1,'Your Reading']],
                             columns=['RPM','Coolant','Oil','Fuel','Condition','Status'])
        df_chart = pd.concat([df_chart, your], ignore_index=True)

    color_map = {'Healthy':'#4ade80','At Risk':'#fbbf24','Critical':'#f87171','Your Reading':'#60a5fa'}

    st.markdown('<div class="section-title">RPM vs Coolant Temp</div>', unsafe_allow_html=True)
    fig1 = px.scatter(df_chart, x='RPM', y='Coolant', color='Status',
                      color_discrete_map=color_map, template='plotly_dark',
                      labels={'RPM':'Engine RPM','Coolant':'Coolant Temp (°C)'})
    fig1.add_hline(y=105, line_dash='dash', line_color='#fbbf24', annotation_text='Warning')
    fig1.add_hline(y=115, line_dash='dash', line_color='#f87171', annotation_text='Critical')
    fig1.update_layout(
        margin=dict(t=10,b=10), height=220,
        paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
        legend=dict(font=dict(color='#8b949e'), bgcolor='rgba(0,0,0,0)'),
        font=dict(color='#8b949e')
    )
    fig1.update_xaxes(gridcolor='#21262d', zerolinecolor='#21262d')
    fig1.update_yaxes(gridcolor='#21262d', zerolinecolor='#21262d')
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<div class="section-title">RPM vs Fuel Pressure</div>', unsafe_allow_html=True)
    fig2 = px.scatter(df_chart, x='RPM', y='Fuel', color='Status',
                      color_discrete_map=color_map, template='plotly_dark',
                      labels={'RPM':'Engine RPM','Fuel':'Fuel Pressure (PSI)'})
    fig2.update_layout(
        margin=dict(t=10,b=10), height=220,
        paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
        legend=dict(font=dict(color='#8b949e'), bgcolor='rgba(0,0,0,0)'),
        font=dict(color='#8b949e')
    )
    fig2.update_xaxes(gridcolor='#21262d', zerolinecolor='#21262d')
    fig2.update_yaxes(gridcolor='#21262d', zerolinecolor='#21262d')
    st.plotly_chart(fig2, use_container_width=True)

# ── HISTORY ────────────────────────────────────────────────────────
st.markdown('<div class="section-title" style="margin-top:1.5rem;">Diagnosis History</div>', unsafe_allow_html=True)

if st.session_state.history:
    color_map_hist = {'🟢  HEALTHY':'#4ade80','🟡  AT RISK':'#fbbf24','🔴  CRITICAL':'#f87171'}
    rows_html = ""
    for h in reversed(st.session_state.history[-10:]):
        c = color_map_hist.get(h['status'], '#8b949e')
        rows_html += f"""
        <div class="hist-row">
            <span class="hist-time">{h['time']}</span>
            <span class="hist-status" style="color:{c};">{h['status']}</span>
            <span class="hist-readings">RPM {h['rpm']} · Cool {h['coolant']}° · Oil {h['oil']} · Fuel {h['fuel']}</span>
            <span class="hist-cost">{h['cost']}</span>
        </div>"""
    st.markdown(f'<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px 14px;">{rows_html}</div>',
                unsafe_allow_html=True)
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.markdown('<div style="color:#484f58;font-size:13px;padding:0.5rem 0;">No diagnoses yet — run one to see history here.</div>',
                unsafe_allow_html=True)
