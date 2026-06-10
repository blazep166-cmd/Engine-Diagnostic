import streamlit as st
import pandas as pd
import plotly.express as px
import pickle, base64, gzip, json
from openai import OpenAI

# ── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Engine AI Diagnostic",
    page_icon="🔧",
    layout="wide"
)

# ── STYLING ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .status-healthy  { background:#0f2a1a; color:#4ade80; border:1px solid #166534; padding:1rem 1.5rem; border-radius:10px; font-size:1.5rem; font-weight:700; margin-bottom:1rem; }
    .status-atrisk   { background:#2a1f0a; color:#fbbf24; border:1px solid #92400e; padding:1rem 1.5rem; border-radius:10px; font-size:1.5rem; font-weight:700; margin-bottom:1rem; }
    .obd-box         { background:#0f1f2a; border:1px solid #1d4e6e; border-radius:8px; padding:0.8rem 1rem; color:#58a6ff; font-size:0.9rem; margin-top:0.5rem; }
    .ai-box          { background:#161b22; border-left:3px solid #58a6ff; border-radius:0 8px 8px 0; padding:1rem 1.2rem; color:#c9d1d9; font-size:0.92rem; line-height:1.7; margin-top:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── API KEY ────────────────────────────────────────────────────────
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    api_key = "YOUR_OPENAI_API_KEY_HERE"

client = OpenAI(api_key=api_key)

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

def get_code_description(code):
    code = code.upper().strip()
    if code in obd_library:
        return obd_library[code]
    return ('Code not found', None)

# ── TRAIN MODEL ────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    training_data = [
        [2500, 90,  35, 55, 0], [3000, 95,  40, 58, 0],
        [1800, 88,  32, 50, 0], [2000, 92,  38, 54, 0],
        [3500, 99,  36, 57, 0], [2800, 93,  41, 56, 0],
        [1500, 85,  30, 52, 0], [2700, 91,  37, 53, 0],
        [2200,107,  22, 42, 1], [4000,110,  18, 38, 1],
        [4500,108,  20, 40, 1], [3800,112,  17, 36, 1],
        [5000,106,  23, 39, 1], [3200,109,  19, 37, 1],
        [600, 120,  10, 28, 2], [550, 125,   8, 25, 2],
        [580, 128,   7, 22, 2], [620, 122,   9, 26, 2],
        [500, 130,   6, 20, 2],
    ]
    from sklearn.ensemble import RandomForestClassifier
    df = pd.DataFrame(training_data, columns=['rpm','coolant','oil','fuel','condition'])
    X  = df[['rpm','coolant','oil','fuel']]
    y  = df['condition']
    m  = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(X, y)
    return m

model = train_model()

# ── HEADER ─────────────────────────────────────────────────────────
st.title("🔧 Engine AI Diagnostic")
st.markdown("Move the sliders to match your engine readings, then click **Run Diagnosis**.")
st.divider()

# ── LAYOUT ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.subheader("Sensor Readings")
    rpm   = st.slider("Engine RPM",          500,  7000, 2500, 50)
    cool  = st.slider("Coolant Temp (°C)",   50,   135,  90,   1)
    oil   = st.slider("Oil Pressure (PSI)",  0,    100,  35,   1)
    fuel  = st.slider("Fuel Pressure (PSI)", 20,   100,  55,   1)

    st.divider()
    st.subheader("Optional")
    obd_code = st.text_input("OBD Code", placeholder="e.g. P0301  (leave blank if none)")
    question = st.text_area("Ask the AI a question",
                             placeholder="e.g. Why is my engine overheating?",
                             height=100)
    run = st.button("Run Diagnosis", type="primary")

with col_right:
    if run:
        new_reading = pd.DataFrame([[rpm, cool, oil, fuel]],
                                    columns=['rpm','coolant','oil','fuel'])
        pred  = model.predict(new_reading)[0]
        proba = model.predict_proba(new_reading)[0]

        status_map = {0:'🟢  HEALTHY', 1:'🟡  AT RISK', 2:'🔴  FAILING'}
        class_map  = {0:'status-healthy', 1:'status-atrisk', 2:'status-atrisk'}
        cost_map   = {0:'$0 – $150', 1:'$150 – $600', 2:'$600 – $3,500+'}
        action_map = {
            0: 'No immediate action needed. Keep up regular maintenance.',
            1: 'Schedule an inspection within 1–2 weeks. Monitor sensors closely.',
            2: 'Stop driving immediately. Urgent inspection required.'
        }

        st.markdown(f'<div class="{class_map[pred]}">{status_map[pred]}</div>',
                    unsafe_allow_html=True)

        st.markdown("**Confidence**")
        st.progress(float(proba[0]), text=f"Healthy  {proba[0]:.0%}")
        if len(proba) > 1:
            st.progress(float(proba[1]), text=f"At Risk  {proba[1]:.0%}")
        if len(proba) > 2:
            st.progress(float(proba[2]), text=f"Failing  {proba[2]:.0%}")

        st.info(action_map[pred])
        st.metric("Estimated Repair Cost", cost_map[pred])

        if obd_code:
            st.markdown("**OBD Code Lookup**")
            desc, system = get_code_description(obd_code)
            if desc != 'Code not found':
                st.markdown(f'<div class="obd-box"><strong>{obd_code.upper()}</strong> — {desc}<br><span style="opacity:0.7">System: {system}</span></div>',
                            unsafe_allow_html=True)
            else:
                st.warning(f"Code {obd_code.upper()} not found in library.")

        if question:
            st.markdown("**AI Diagnosis**")
            with st.spinner("Thinking..."):
                obd_info = obd_library.get(obd_code.upper().strip(), (None, None)) if obd_code else (None, None)
                prompt = f"""
Analyzing engine diagnostics.
Sensor readings: RPM={rpm}, Coolant={cool}C, Oil Pressure={oil}PSI, Fuel Pressure={fuel}PSI.
Predicted condition: {status_map[pred]}
OBD Code: {obd_code if obd_code else 'None'}
Code description: {obd_info[0]}
System: {obd_info[1]}
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
            st.markdown(f'<div class="ai-box">{answer}</div>', unsafe_allow_html=True)

    else:
        st.markdown("### 👈 Set your readings and click Run Diagnosis")
        st.markdown("Your results and charts will appear here.")

    # Charts always visible
    st.divider()
    st.subheader("Sensor Reference Charts")

    training_data = [
        [2500,90,35,55,0],[3000,95,40,58,0],[1800,88,32,50,0],
        [2000,92,38,54,0],[3500,99,36,57,0],[2800,93,41,56,0],
        [2200,107,22,42,1],[4000,110,18,38,1],[4500,108,20,40,1],
        [600,120,10,28,2],[550,125,8,25,2],[580,128,7,22,2],
    ]
    df_chart = pd.DataFrame(training_data, columns=['RPM','Coolant','Oil','Fuel','Condition'])
    df_chart['Status'] = df_chart['Condition'].map({0:'Healthy',1:'At Risk',2:'Failing'})
    if run:
        your_row = pd.DataFrame([[rpm,cool,oil,fuel,-1,'Your Reading']],
                                  columns=['RPM','Coolant','Oil','Fuel','Condition','Status'])
        df_chart = pd.concat([df_chart, your_row], ignore_index=True)

    color_map = {'Healthy':'green','At Risk':'orange','Failing':'red','Your Reading':'blue'}

    fig1 = px.scatter(df_chart, x='RPM', y='Coolant', color='Status',
                      color_discrete_map=color_map,
                      title='RPM vs Coolant Temp',
                      template='plotly_dark')
    fig1.add_hline(y=105, line_dash='dash', line_color='orange', annotation_text='Warning')
    fig1.add_hline(y=115, line_dash='dash', line_color='red',    annotation_text='Danger')
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df_chart, x='RPM', y='Fuel', color='Status',
                      color_discrete_map=color_map,
                      title='RPM vs Fuel Pressure',
                      template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)
