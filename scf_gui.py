import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ===================== Page =====================
st.set_page_config(
    page_title="SCF Prediction System for Deck–U-rib Double-sided Welded Joints",
    page_icon="🔧",
    layout="wide"
)

# ===================== Times New Roman =====================
st.markdown(
    """
    <style>
    body, p, div, span, label, input, textarea, button {
        font-family: "Times New Roman" !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: "Times New Roman" !important;
    }

    .param-label {
        font-size: 15px;
        margin-bottom: 2px;
        line-height: 1.2;
    }

    .target-name {
        font-size: 18px;
        font-weight: 600;
        margin-top: 8px;
        margin-bottom: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== Label constants =====================
SCF_EX_LABEL = "External weld toe SCFₑₙₛ₋ₑₓ"
SCF_IN_LABEL = "Internal weld toe SCFₑₙₛ₋ᵢₙ"


# ===================== Load models =====================
@st.cache_resource
def load_models():
    model_yw3 = joblib.load("model_yw3.pkl")
    model_yn3 = joblib.load("model_yn3.pkl")
    return model_yw3, model_yn3


# ===================== SCFENS-ex =====================
def predict_yw3(model_package, features):
    base_names = model_package["base_names"]
    base_models = model_package["base_models"]
    meta_model = model_package["meta_model"]
    feature_cols = model_package["feature_cols"]

    X_df = pd.DataFrame(
        features.reshape(1, -1),
        columns=feature_cols
    )

    meta_features = []

    for name in base_names:
        pred = base_models[name].predict(X_df)
        pred = np.asarray(pred).reshape(1, -1)
        meta_features.extend(pred[0].tolist())

    meta_features = np.array(meta_features).reshape(1, -1)
    scfens_ex = meta_model.predict(meta_features)[0]

    return scfens_ex


# ===================== SCFENS-in =====================
def predict_yn3(model_package, features):
    xgb_model = model_package["XGB"]
    gbdt_model = model_package["GBDT"]
    feature_cols = model_package["feature_cols"]
    output_index = model_package.get("output_index", 1)

    X_df = pd.DataFrame(
        features.reshape(1, -1),
        columns=feature_cols
    )

    pred_xgb = np.asarray(
        xgb_model.predict(X_df)
    ).reshape(1, -1)

    pred_gbdt = np.asarray(
        gbdt_model.predict(X_df)
    ).reshape(1, -1)

    scfens_in = (
        pred_xgb[0, output_index]
        +
        pred_gbdt[0, output_index]
    ) / 2

    return scfens_in


# ===================== Title =====================
st.title(
    "🔧 SCF Prediction System for Deck–U-rib Double-sided Welded Joints"
)


# ===================== Sidebar =====================
st.sidebar.header(
    "📘 Model Information"
)

st.sidebar.markdown(
    """
    <div style="
    background-color:#e8f1ff;
    padding:14px;
    border-radius:8px;
    line-height:1.6;">

    <b>Prediction Channels:</b><br>

    External weld toe:
    CAT+GBDT+ANN Stacking<br>

    Internal weld toe:
    XGB+GBDT simple averaging<br><br>

    <b>Prediction Targets:</b><br>

    SCF<sub>ENS-ex</sub>:
    External weld toe SCF<br>

    SCF<sub>ENS-in</sub>:
    Internal weld toe SCF

    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🎯 Prediction Channel"
)

target = st.sidebar.radio(
    "Select prediction target",
    [
        SCF_EX_LABEL,
        SCF_IN_LABEL
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Developed by College of Civil Engineering, Nanjing Forestry University"
)


# ===================== Main =====================
st.subheader(
    "Geometric Parameters and Specimen Diagram"
)

input_col, fig_col = st.columns(
    [0.8, 1.7]
)


with input_col:

    st.markdown(
        "**Geometric Parameters**"
    )

    p1, p2 = st.columns(2)

    with p1:

        st.markdown(
            '<div class="param-label"><i>X</i><sub>1</sub> Deck thickness</div>',
            unsafe_allow_html=True
        )
        X1 = st.number_input(
            label="",
            min_value=12.0,
            max_value=18.0,
            value=14.0,
            step=0.5,
            key="x1",
            label_visibility="collapsed"
        )

        st.markdown(
            '<div class="param-label"><i>X</i><sub>2</sub> U-rib thickness</div>',
            unsafe_allow_html=True
        )
        X2 = st.number_input(
            label="",
            min_value=6.0,
            max_value=10.0,
            value=8.0,
            step=0.5,
            key="x2",
            label_visibility="collapsed"
        )

        st.markdown(
            '<div class="param-label"><i>X</i><sub>3</sub> U-rib height</div>',
            unsafe_allow_html=True
        )
        X3 = st.number_input(
            label="",
            min_value=200.0,
            max_value=280.0,
            value=240.0,
            step=10.0,
            key="x3",
            label_visibility="collapsed"
        )

        st.markdown(
            '<div class="param-label"><i>X</i><sub>4</sub> U-rib width</div>',
            unsafe_allow_html=True
        )
        X4 = st.number_input(
            label="",
            min_value=165.0,
            max_value=181.0,
            value=173.0,
            step=2.0,
            key="x4",
            label_visibility="collapsed"
        )

    with p2:

        st.markdown(
            '<div class="param-label"><i>X</i><sub>5</sub> External deck-side weld toe length</div>',
            unsafe_allow_html=True
        )
        X5 = st.number_input(
            label="",
            min_value=6.0,
            max_value=9.0,
            value=7.0,
            step=0.5,
            key="x5",
            label_visibility="collapsed"
        )

        st.markdown(
            '<div class="param-label"><i>X</i><sub>6</sub> External U-rib-side weld toe length</div>',
            unsafe_allow_html=True
        )
        X6 = st.number_input(
            label="",
            min_value=6.0,
            max_value=9.0,
            value=7.0,
            step=0.5,
            key="x6",
            label_visibility="collapsed"
        )

        st.markdown(
            '<div class="param-label"><i>X</i><sub>7</sub> Internal deck-side weld toe length</div>',
            unsafe_allow_html=True
        )
        X7 = st.number_input(
            label="",
            min_value=6.0,
            max_value=9.0,
            value=7.0,
            step=0.5,
            key="x7",
            label_visibility="collapsed"
        )

        st.markdown(
            '<div class="param-label"><i>X</i><sub>8</sub> Internal U-rib-side weld toe length</div>',
            unsafe_allow_html=True
        )
        X8 = st.number_input(
            label="",
            min_value=6.0,
            max_value=9.0,
            value=7.0,
            step=0.5,
            key="x8",
            label_visibility="collapsed"
        )

    predict_btn = st.button(
        "🚀 Predict",
        type="primary",
        width="stretch"
    )


    # ===================== Result =====================
    if predict_btn:

        features_mm = np.array(
            [X1, X2, X3, X4, X5, X6, X7, X8]
        )

        features = features_mm / 1000

        model_yw3, model_yn3 = load_models()

        st.markdown(
            "### Prediction Result"
        )

        st.success(
            "✅ Prediction completed"
        )

        if target == SCF_EX_LABEL:

            result = predict_yw3(
                model_yw3,
                features
            )

            st.markdown(
                '<div class="target-name">SCF<sub>ENS-ex</sub></div>',
                unsafe_allow_html=True
            )

            st.metric(
                label="",
                value=f"{result:.4f}",
                label_visibility="collapsed"
            )

        else:

            result = predict_yn3(
                model_yn3,
                features
            )

            st.markdown(
                '<div class="target-name">SCF<sub>ENS-in</sub></div>',
                unsafe_allow_html=True
            )

            st.metric(
                label="",
                value=f"{result:.4f}",
                label_visibility="collapsed"
            )

        show_input = st.checkbox(
            "View Input Parameters"
        )

        if show_input:
            input_df = pd.DataFrame({

                "Parameter": [
                    "X₁", "X₂", "X₃", "X₄",
                    "X₅", "X₆", "X₇", "X₈"
                ],

                "Value (mm)": [
                    X1, X2, X3, X4,
                    X5, X6, X7, X8
                ]

            })

            st.dataframe(
                input_df,
                hide_index=True
            )


with fig_col:

    st.markdown(
        "**Specimen Diagram**"
    )

    if os.path.exists(
        "specimen.png"
    ):

        st.image(
            "specimen.png",
            use_container_width=True
        )

    else:

        st.warning(
            "specimen.png not found"
        )


st.caption(
    "📌 Prediction results are intended for engineering analysis and research reference only."
)
