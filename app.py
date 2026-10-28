```python
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Predictive Maintenance System")
st.subheader("Machine Failure Risk Prediction")

st.write(
    "Enter the operating conditions of a machine to predict "
    "whether machine failure is likely."
)

model_path = "models/best_model.joblib"

if not os.path.exists(model_path):
    st.error("Trained model file is missing. Please add the model file.")
    st.stop()

model = joblib.load(model_path)

st.header("Enter Machine Parameters")

air_temp = st.number_input(
    "Air temperature [K]", min_value=250.0, max_value=350.0, value=300.0
)

process_temp = st.number_input(
    "Process temperature [K]", min_value=250.0, max_value=400.0, value=310.0
)

rot_speed = st.number_input(
    "Rotational speed [rpm]", min_value=0, max_value=10000, value=1500
)

torque = st.number_input(
    "Torque [Nm]", min_value=0.0, max_value=500.0, value=40.0
)

tool_wear = st.number_input(
    "Tool wear [min]", min_value=0, max_value=500, value=100
)

if st.button("Predict Machine Failure"):
    input_data = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        failure_probability = probabilities[list(model.classes_).index(1)] * 100
    else:
        failure_probability = None

    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted")
    else:
        st.success("✅ No Machine Failure Predicted")

    if failure_probability is not None:
        st.metric("Estimated Failure Probability", f"{failure_probability:.2f}%")

st.caption(
    "This prediction is based on a machine-learning model trained "
    "using the AI4I 2020 Predictive Maintenance Dataset. "
    "It is an educational prediction tool, not a guarantee of machine safety."
)
```
