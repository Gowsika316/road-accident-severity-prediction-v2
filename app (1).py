import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor

model = joblib.load("accident_severity_model_compressed.pkl")
preprocessor = joblib.load("accident_preprocessor (1).pkl")

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Road Accident Severity Prediction")
st.write("Enter the accident, road, and environmental details.")

with st.form("accident_form"):
    st.subheader("Accident Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        vehicles = st.number_input(
            "Number of Vehicles", 1, 20, 2
        )
        casualties = st.number_input(
            "Number of Casualties", 1, 30, 1
        )
        speed = st.selectbox(
            "Speed Limit", [10, 20, 30, 40, 50, 60, 70], index=2
        )
        day = st.selectbox(
            "Day of Week", [1, 2, 3, 4, 5, 6, 7], index=3
        )
        road1 = st.selectbox(
            "1st Road Class", [1, 2, 3, 4, 5, 6], index=2
        )
        road_type = st.selectbox(
            "Road Type", [1, 2, 3, 6, 7, 9], index=0
        )

    with c2:
        junction_detail = st.selectbox(
            "Junction Detail", [-1, 0, 1, 2, 3, 5, 6, 7, 8, 9]
        )
        junction_control = st.selectbox(
            "Junction Control", [-1, 0, 1, 2, 3, 4]
        )
        road2 = st.selectbox(
            "2nd Road Class", [-1, 0, 1, 2, 3, 4, 5, 6]
        )
        pedestrian_human = st.selectbox(
            "Pedestrian Crossing - Human Control", [-1, 0, 1, 2]
        )
        pedestrian_physical = st.selectbox(
            "Pedestrian Crossing - Physical Facilities",
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        )
        urban_rural = st.selectbox(
            "Urban or Rural Area", [1, 2]
        )

    with c3:
        light = st.selectbox(
            "Light Conditions", [1, 4, 5, 6, 7]
        )
        weather = st.selectbox(
            "Weather Conditions", [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        surface = st.selectbox(
            "Road Surface Conditions", [-1, 1, 2, 3, 4, 5]
        )
        special = st.selectbox(
            "Special Conditions at Site", [-1, 0, 1, 2, 3, 4, 5, 6, 7]
        )
        hazards = st.selectbox(
            "Carriageway Hazards", [-1, 0, 1, 2, 3, 4, 5, 6, 7]
        )
        police = st.selectbox(
            "Did Police Officer Attend Scene?", [-1, 1, 2, 3]
        )

    submitted = st.form_submit_button(
        "Predict Accident Severity",
        use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "Number_of_Vehicles": vehicles,
        "Number_of_Casualties": casualties,
        "Day_of_Week": day,
        "1st_Road_Class": road1,
        "Road_Type": road_type,
        "Speed_limit": speed,
        "Junction_Detail": junction_detail,
        "Junction_Control": junction_control,
        "2nd_Road_Class": road2,
        "Pedestrian_Crossing-Human_Control": pedestrian_human,
        "Pedestrian_Crossing-Physical_Facilities": pedestrian_physical,
        "Light_Conditions": light,
        "Weather_Conditions": weather,
        "Road_Surface_Conditions": surface,
        "Special_Conditions_at_Site": special,
        "Carriageway_Hazards": hazards,
        "Urban_or_Rural_Area": urban_rural,
        "Did_Police_Officer_Attend_Scene_of_Accident": police
    }])

    processed = preprocessor.transform(input_data)
    prediction = int(model.predict(processed)[0])

    st.subheader("Prediction Result")
    st.metric("Predicted Severity Class", prediction)

    st.info(
        "This is an academic prototype. Confirm the dataset codebook "
        "before interpreting severity class codes."
    )

    probabilities = model.predict_proba(processed)[0]

    results = pd.DataFrame({
        "Severity Class": model.classes_.astype(int),
        "Model Probability (%)": (probabilities * 100).round(2)
    })

    st.subheader("Model Probability Estimates")
    st.dataframe(results, use_container_width=True, hide_index=True)
