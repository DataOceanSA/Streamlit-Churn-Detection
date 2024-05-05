import streamlit as st
import pickle
import numpy as np
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor_loaded = data["model"]
Region = data["region"]
tenure_encoder = data["tenure"]
top = data["top"]


def show_predict_page():
    st.title("Churn Prediction")

    st.write("""### We need some information to predict the churn status""")
    default_values = [14950.0, 44.0, 15150.0, 5050.0, 52.0, 0.0, 141.0, 316.0, 151.0, 59.0, "All-net 500F=2000F;5d", 23.0]

    regions_list = ('FATICK', 'DAKAR', 'LOUGA', 'TAMBACOUNDA', 'KAOLACK', 'SAINT-LOUIS', 'THIES', 'KAFFRINE', 'DIOURBEL', 'MATAM', 'KOLDA', 'ZIGUINCHOR', 'SEDHIOU', 'KEDOUGOU')
    tenure_values = ('K > 24 month', 'H 15-18 month', 'I 18-21 month', 'F 9-12 month', 'G 12-15 month', 'J 21-24 month', 'E 6-9 month', 'D 3-6 month')
    region = st.selectbox("Region", regions_list)
    tenure = st.selectbox("tenure", tenure_values)

    # Affichage des champs de saisie avec les valeurs par défaut
    montant = st.number_input("MONTANT", value=default_values[0])
    frequence_rech = st.number_input("FREQUENCE_RECH", value=default_values[1])
    revenue = st.number_input("REVENUE", value=default_values[2])
    arpu_segment = st.number_input("ARPU_SEGMENT", value=default_values[3])
    frequence = st.number_input("FREQUENCE", value=default_values[4])
    data_volume = st.number_input("DATA_VOLUME", value=default_values[5])
    on_net = st.number_input("ON_NET", value=default_values[6])
    orange = st.number_input("ORANGE", value=default_values[7])
    tigo = st.number_input("TIGO", value=default_values[8])
    regularity = st.number_input("REGULARITY", value=default_values[9])
    top_pack = st.text_input("TOP_PACK", value=default_values[10])
    freq_top_pack = st.number_input("FREQ_TOP_PACK", value=default_values[11])
    ok = st.button("Check")
    if ok:
        data_np = np.array([[region, tenure, montant,frequence_rech,revenue,arpu_segment,frequence,data_volume,on_net,orange,tigo,regularity,top_pack,freq_top_pack]])
        data_np[:, 0] = Region.transform(data_np[:,0])
        data_np[:, 1] = tenure_encoder.transform(data_np[:,1])
        data_np[:, -2] = top.transform(data_np[:,-2])
        data_np=data_np.astype(float)
        churn=regressor_loaded.predict(data_np)
        result = "Churn" if churn[0] == 1 else "Not Churn"
        st.write(f"Prediction Result: {result}")

show_predict_page()