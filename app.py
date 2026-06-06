import os
import numpy as np
import torch
import streamlit as st

import train as _train_module
_train_module.device = torch.device('cpu')  # force CPU for portable inference

# ---------------------------------------------------------------------------
# Metadata: label, type, units, valid range for each NHANES variable
# ---------------------------------------------------------------------------
VARIABLE_META = {
    'RIDAGEYR': {'label': 'Age',                   'unit': 'years',  'type': 'int',   'min': 18,   'max': 85,    'step': 1,   'default': 45},
    'RIAGENDR': {'label': 'Gender',                'unit': '',       'type': 'cat',   'options': {1: 'Male', 2: 'Female'}},
    'BMXHT':    {'label': 'Height',                'unit': 'cm',     'type': 'float', 'min': 100.0,'max': 220.0, 'step': 0.1, 'default': 170.0},
    'BMXWT':    {'label': 'Weight',                'unit': 'kg',     'type': 'float', 'min': 20.0, 'max': 300.0, 'step': 0.1, 'default': 80.0},
    'BMXBMI':   {'label': 'BMI',                   'unit': 'kg/m²',  'type': 'float', 'min': 10.0, 'max': 80.0,  'step': 0.1, 'default': 27.5},
    'BMXWAIST': {'label': 'Waist circumference',   'unit': 'cm',     'type': 'float', 'min': 40.0, 'max': 200.0, 'step': 0.1, 'default': 95.0},
    'BPXDI1':   {'label': 'Diastolic BP',          'unit': 'mmHg',   'type': 'float', 'min': 0.0,  'max': 130.0, 'step': 1.0, 'default': 75.0},
    'BPXSY1':   {'label': 'Systolic BP',           'unit': 'mmHg',   'type': 'float', 'min': 60.0, 'max': 250.0, 'step': 1.0, 'default': 120.0},
    'BPXPLS':   {'label': 'Pulse',                 'unit': 'bpm',    'type': 'float', 'min': 20.0, 'max': 220.0, 'step': 1.0, 'default': 72.0},
    'LBXSCH':   {'label': 'Total cholesterol',     'unit': 'mg/dL',  'type': 'float', 'min': 50.0, 'max': 600.0, 'step': 1.0, 'default': 200.0},
    'LBXSTR':   {'label': 'Triglycerides',         'unit': 'mg/dL',  'type': 'float', 'min': 10.0, 'max': 2500.0,'step': 1.0, 'default': 150.0},
    'LBXGH':    {'label': 'HbA1c',                 'unit': '%',      'type': 'float', 'min': 2.0,  'max': 20.0,  'step': 0.1, 'default': 5.5},
    'LBXGLU':   {'label': 'Fasting glucose',       'unit': 'mg/dL',  'type': 'float', 'min': 40.0, 'max': 800.0, 'step': 1.0, 'default': 95.0},
}

COMBINATIONS = {
    'Comb 1 — Age + Gender (2 vars)':                                        ['RIDAGEYR', 'RIAGENDR'],
    'Comb 2 — Anthropometrics + BP (7 vars)':                                ['BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS'],
    'Comb 3 — Age + Gender + Anthropometrics + BP (9 vars)':                 ['RIDAGEYR', 'RIAGENDR', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS'],
    'Comb 4 — Comb 3 + Triglycerides (10 vars)':                            ['RIDAGEYR', 'RIAGENDR', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBXSTR'],
    'Comb 5 — Comb 4 + Cholesterol + HbA1c (12 vars)':                      ['RIDAGEYR', 'RIAGENDR', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBXSCH', 'LBXSTR', 'LBXGH'],
    'Comb 6 — Comb 4 + Cholesterol + Glucose (12 vars)':                    ['RIDAGEYR', 'RIAGENDR', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBXSCH', 'LBXSTR', 'LBXGLU'],
    'Comb 7 — Comb 4 + Cholesterol + Glucose + HbA1c (13 vars)':            ['RIDAGEYR', 'RIAGENDR', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBXSCH', 'LBXSTR', 'LBXGLU', 'LBXGH'],
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code', 'models')


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def load_bundle(comb_idx: int):
    path = os.path.join(MODELS_DIR, f'predictor_comb_{comb_idx}.pt')
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    bundle['predictor']._model_ce = bundle['predictor']._model_ce.cpu()
    bundle['predictor']._model_rr = bundle['predictor']._model_rr.cpu()
    return bundle

# ---------------------------------------------------------------------------
# Preprocessing a single user input into a model-ready array
# ---------------------------------------------------------------------------

def build_feature_vector(comb: list, user_values: dict) -> np.ndarray:
    cat_variables = ['RIAGENDR']
    parts = []
    for var in comb:
        if var in cat_variables:
            val = user_values[var]   # 1 or 2
            # One-hot: categories sorted = [1, 2], so col0=Male, col1=Female
            oh = np.array([[1.0, 0.0]] if val == 1 else [[0.0, 1.0]], dtype='float32')
            parts.append(oh)
        else:
            raw = np.array([[float(user_values[var])]], dtype='float32')
            parts.append(raw)
    return np.hstack(parts)  # shape (1, n_features)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title='Metabolic Syndrome Predictor', layout='centered')
st.title('Metabolic Syndrome Predictor')
st.caption('Conformal inference model · NHANES dataset')

comb_name = st.selectbox('Select feature combination', list(COMBINATIONS.keys()))
comb_idx  = list(COMBINATIONS.keys()).index(comb_name)
comb      = COMBINATIONS[comb_name]

st.divider()
st.subheader('Patient data')

user_values = {}
cols = st.columns(2)
col_idx = 0

for var in comb:
    meta = VARIABLE_META[var]
    label = f"{meta['label']}" + (f" ({meta['unit']})" if meta['unit'] else '')
    with cols[col_idx % 2]:
        if meta['type'] == 'cat':
            options = meta['options']
            choice  = st.selectbox(label, options=list(options.keys()),
                                   format_func=lambda k, o=options: o[k], key=var)
            user_values[var] = choice
        elif meta['type'] == 'int':
            user_values[var] = st.number_input(
                label, min_value=meta['min'], max_value=meta['max'],
                step=meta['step'], value=meta['default'], key=var
            )
        else:
            user_values[var] = st.number_input(
                label, min_value=float(meta['min']), max_value=float(meta['max']),
                step=float(meta['step']), value=float(meta['default']), key=var,
                format='%.1f'
            )
    col_idx += 1

st.divider()

if st.button('Predict', type='primary', use_container_width=True):
    with st.spinner('Loading model…'):
        bundle    = load_bundle(comb_idx)
        predictor = bundle['predictor']
        scaler    = bundle['scaler']

    x_raw = build_feature_vector(comb, user_values)
    x_scaled = scaler.transform(x_raw).astype('float32')

    test_data = {
        'x': x_scaled,
        'y': np.zeros((1, 1), dtype='float32'),
        'w': np.ones((1, 1),  dtype='float32'),
        'z': np.zeros((1, 1), dtype='float32'),
    }

    ci_result, _ = predictor.classify(test_data)

    prob       = float(ci_result['p'][0])
    prediction = '🔴 Positive' if prob >= 0.5 else '🟢 Negative'

    st.subheader('Result')
    c1, c2 = st.columns(2)
    with c1:
        st.metric('Probability of metabolic syndrome', f'{prob:.1%}')
    with c2:
        st.metric('Prediction', prediction)
