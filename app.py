import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from openai import OpenAI

# ── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Engine AI Diagnostic",
    page_icon="🔧",
    layout="wide"
)

# ── CUSTOM STYLING ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 0.04em;
    }
    .main { background-color: #0d1117; }

    .status-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .healthy  { background: #0f2a1a; color: #4ade80; border: 1px solid #166534; }
    .atrisk   { background: #2a1f0a; color: #fbbf24; border: 1px solid #92400e; }
    .failing  { background: #2a0a0a; color: #f87171; border: 1px solid #991b1b; }

    .metric-row {
        display: flex;
        gap: 12px;
        margin-bottom: 1rem;
    }
    .metric-card {
        flex: 1;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #e6edf3;
    }
    .section-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8b949e;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
        margin: 1.2rem 0 0.8rem;
    }
    .obd-found {
        background: #0f1f2a;
        border: 1px solid #1d4e6e;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        color: #58a6ff;
        font-size: 0.9rem;
    }
    .ai-answer {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        color: #c9d1d9;
        font-size: 0.92rem;
        line-height: 1.7;
        margin-top: 0.5rem;
    }
    .stSlider > div > div { background: #21262d; }
    div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
    .stButton > button {
        background: #238636;
        color: white;
        border: none;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 0.6rem 2rem;
        border-radius: 6px;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #2ea043; }
</style>
""", unsafe_allow_html=True)

# ── API KEY ────────────────────────────────────────────────────────
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
client = OpenAI(api_key=OPENAI_API_KEY)

# ── OBD LIBRARY ────────────────────────────────────────────────────
obd_library = {
    'P0217': ('Engine Over Temperature',           'Engine Cooling System'),
    'P0128': ('Coolant Temp Below Thermostat',      'Engine Cooling System'),
    'P0520': ('Oil Pressure Sensor Circuit',        'Lubrication System'),
    'P0521': ('Oil Pressure Sensor Range/Perf',     'Lubrication System'),
    'P0087': ('Fuel Rail Pressure Too Low',         'Fuel System'),
    'P0088': ('Fuel Rail Pressure Too High',        'Fuel System'),
    'P0171': ('System Too Lean Bank 1',             'Fuel System'),
    'P0172': ('System Too Rich Bank 1',             'Fuel System'),
    'P0301': ('Cylinder 1 Misfire Detected',        'Ignition System'),
    'P0302': ('Cylinder 2 Misfire Detected',        'Ignition System'),
    'P0303': ('Cylinder 3 Misfire Detected',        'Ignition System'),
    'P0304': ('Cylinder 4 Misfire Detected',        'Ignition System'),
    'P0420': ('Catalyst Efficiency Below Threshold','Emission System'),
    'P0113': ('Intake Air Temp Sensor High Input',  'Air Intake System'),
    'P0136': ('O2 Sensor Circuit Fault',            'Emission System'),
}

# ── TRAIN MODEL (cached so it only runs once) ──────────────────────
@st.cache_resource
def train_model():
    training_data = [
        [2500, 90,  35, 55, 0],
        [3000, 95,  40, 58, 0],
        [1800, 88,  32, 50, 0],
        [2000, 92,  38, 54, 0],
        [3500, 99,  36, 57, 0],
        [2800, 93,  41, 56, 0],
        [1500, 85,  30, 52, 0],
        [2700, 91,  37, 53, 0],
        [2200, 107, 22, 42, 1],
        [4000, 110, 18, 38, 1],
        [4500, 108, 20, 40, 1],
        [3800, 112, 17, 36, 1],
        [5000, 106, 23, 39, 1],
        [3200, 109, 19, 37, 1],
        [600,  120, 10, 28, 2],
        [550,  125,  8, 25, 2],
        [580,  128,  7, 22, 2],
        [620,  122,  9, 26, 2],
        [500,  130,  6, 20, 2],
    ]
    df = pd.DataFrame(training_data, columns=['rpm','coolant','oil','fuel','condition'])
    X  = df[['rpm','coolant','oil','fuel']]
    y  = df['condition']
    m  = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(X, y)
    return m, df

model, df_train = train_model()

# ── HEADER ─────────────────────────────────────────────────────────
st.markdown("# 🔧 Engine AI Diagnostic")
st.markdown("Move the sliders to match your engine readings, then click **Run Diagnosis**.")
st.divider()

# ── LAYOUT: sidebar = inputs, main = results ───────────────────────
with st.sidebar:
    st.markdown("## Sensor Readings")

    rpm   = st.slider("Engine RPM",          min_value=500,  max_value=7000, value=2500, step=50)
    cool  = st.slider("Coolant Temp (°C)",   min_value=50,   max_value=135,  value=90,   step=1)
    oil   = st.slider("Oil Pressure (PSI)",  min_value=0,    max_value=100,  value=35,   step=1)
    fuel  = st.slider("Fuel Pressure (PSI)", min_value=20,   max_value=100,  value=55,   step=1)

    st.markdown("---")
    st.markdown("## Optional")

    obd_code = st.text_input("OBD Code", placeholder="e.g. P0301  (leave blank if none)")
    question = st.text_area("Ask the AI a question",
                             placeholder="e.g. Why is my engine overheating?",
                             height=100)

    run = st.button("Run Diagnosis")

# ── RESULTS ────────────────────────────────────────────────────────
col_results, col_charts = st.columns([1, 1.4], gap="large")

with col_results:
    if run:
        # Prediction
        new_reading = pd.DataFrame([[rpm, cool, oil, fuel]],
                                    columns=['rpm','coolant','oil','fuel'])
        pred  = model.predict(new_reading)[0]
        proba = model.predict_proba(new_reading)[0]

        status_map = {0: '🟢  HEALTHY',    1: '🟡  AT RISK',    2: '🔴  FAILING'}
        class_map  = {0: 'healthy',         1: 'atrisk',         2: 'failing'}
        cost_map   = {0: '$0 – $150',       1: '$150 – $600',    2: '$600 – $3,500+'}
        action_map = {
            0: 'No immediate action needed. Keep up regular maintenance.',
            1: 'Schedule an inspection within 1–2 weeks. Monitor sensors closely.',
            2: 'Stop driving immediately. Urgent inspection required.'
        }

        st.markdown(f'<div class="status-box {class_map[pred]}">{status_map[pred]}</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="section-header">Confidence</div>', unsafe_allow_html=True)
        st.progress(float(proba[0]), text=f"Healthy  {proba[0]:.0%}")
        st.progress(float(proba[1]), text=f"At Risk  {proba[1]:.0%}")
        st.progress(float(proba[2]), text=f"Failing  {proba[2]:.0%}")

        st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Est. Repair Cost</div>
                <div class="metric-value">{cost_map[pred]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info(action_map[pred])

        # OBD Code
        if obd_code:
            st.markdown('<div class="section-header">OBD Code Lookup</div>', unsafe_allow_html=True)
            code_up = obd_code.upper().strip()
            if code_up in obd_library:
                desc, system = obd_library[code_up]
                st.markdown(f"""
                <div class="obd-found">
                    <strong>{code_up}</strong> — {desc}<br>
                    <span style="color:#8b949e;">System: {system}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.warning(f"Code {code_up} not found in library.")

        # AI Answer
        if question:
            st.markdown('<div class="section-header">AI Diagnosis</div>', unsafe_allow_html=True)
            with st.spinner("Thinking..."):
                obd_desc = obd_library.get(obd_code.upper().strip(), (None, None))[0] if obd_code else None
                obd_sys  = obd_library.get(obd_code.upper().strip(), (None, None))[1] if obd_code else None

                prompt = f"""
Analyzing engine diagnostics.

Sensor readings:
- Engine RPM: {rpm}
- Coolant Temp: {cool}°C
- Oil Pressure: {oil} PSI
- Fuel Pressure: {fuel} PSI
- Predicted condition: {status_map[pred]}

OBD Code: {obd_code if obd_code else 'None'}
Description: {obd_desc}
System: {obd_sys}

User question: {question}
"""
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert automotive diagnostic technician. Give clear, practical, plain-English advice."},
                        {"role": "user",   "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content

            st.markdown(f'<div class="ai-answer">{answer}</div>', unsafe_allow_html=True)

    else:
        st.markdown("#### 👈 Set your readings and click Run Diagnosis")
        st.markdown("Your results will appear here.")

# ── CHARTS ─────────────────────────────────────────────────────────
with col_charts:
    def get_status(c):
        return {0: 'Healthy', 1: 'At Risk', 2: 'Failing'}[c]

    df_display = df_train.copy()
    df_display['Status'] = df_display['condition'].apply(get_status)

    if run:
        your_row = pd.DataFrame([[rpm, cool, oil, fuel, -1, 'Your Reading']],
                                  columns=['rpm','coolant','oil','fuel','condition','Status'])
        df_display = pd.concat([df_display, your_row], ignore_index=True)

    color_map = {'Healthy':'#4ade80','At Risk':'#fbbf24','Failing':'#f87171','Your Reading':'#60a5fa'}

    st.markdown('<div class="section-header">RPM vs Coolant Temperature</div>', unsafe_allow_html=True)
    fig1 = px.scatter(df_display, x='rpm', y='coolant', color='Status',
                      color_discrete_map=color_map,
                      labels={'rpm':'Engine RPM','coolant':'Coolant Temp (°C)'},
                      template='plotly_dark')
    fig1.add_hline(y=105, line_dash='dash', line_color='#fbbf24', annotation_text='Warning 105°C')
    fig1.add_hline(y=115, line_dash='dash', line_color='#f87171', annotation_text='Danger 115°C')
    fig1.update_layout(margin=dict(t=10,b=10), height=240, legend_title_text='')
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<div class="section-header">RPM vs Fuel Pressure</div>', unsafe_allow_html=True)
    fig2 = px.scatter(df_display, x='rpm', y='fuel', color='Status',
                      color_discrete_map=color_map,
                      labels={'rpm':'Engine RPM','fuel':'Fuel Pressure (PSI)'},
                      template='plotly_dark')
    fig2.update_layout(margin=dict(t=10,b=10), height=240, legend_title_text='')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Oil Pressure Distribution</div>', unsafe_allow_html=True)
    fig3 = px.histogram(df_display[df_display['Status'] != 'Your Reading'],
                        x='oil', color='Status',
                        color_discrete_map=color_map,
                        labels={'oil':'Oil Pressure (PSI)'},
                        template='plotly_dark', barmode='overlay')
    fig3.update_layout(margin=dict(t=10,b=10), height=220, legend_title_text='')
    st.plotly_chart(fig3, use_container_width=True)
