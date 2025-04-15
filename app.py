import streamlit as st
import pandas as pd
import joblib
import os
from utils import load_data, transform_new_data

def main():
    st.title("Konut Fiyat Tahmin Uygulaması")

    raw_df = load_data('HouseData2.xlsx')
    if raw_df is None:
        st.error("Veri yüklenirken bir hata oluştu. Lütfen dosya yolunu ve dosya içeriğini kontrol edin.")
        st.stop()

    # İlçe ve mahalle listesi oluşturma
    if ('ilce' in raw_df.columns) and ('mahalle' in raw_df.columns):
        raw_df['ilce'] = raw_df['ilce'].fillna("Bilinmiyor")
        raw_df['mahalle'] = raw_df['mahalle'].fillna("Bilinmiyor")
        ilce_list = sorted(raw_df['ilce'].unique(), key=lambda x: str(x))
        ilce_mahalle_map = {
            ilce: sorted(raw_df.loc[raw_df['ilce'] == ilce, 'mahalle'].unique(), key=lambda x: str(x))
            for ilce in ilce_list
        }
    else:
        ilce_list = []
        ilce_mahalle_map = {}

    st.sidebar.header("Konut Özellikleri Seçimi")

    selected_ilce = st.sidebar.selectbox("İlçe Seçiniz", ilce_list)
    if selected_ilce in ilce_mahalle_map:
        selected_mahalle = st.sidebar.selectbox("Mahalle Seçiniz", ilce_mahalle_map[selected_ilce])
    else:
        selected_mahalle = "Bilinmiyor"

    selected_tip = st.sidebar.selectbox(
        "Bina Tipi Seçiniz",
        sorted(raw_df['tip'].dropna().unique(), key=lambda x: str(x)) if 'tip' in raw_df.columns else []
    )

    selected_isitma = st.sidebar.selectbox(
        "Isıtma Türü Seçiniz",
        sorted(raw_df['isitma'].dropna().unique(), key=lambda x: str(x)) if 'isitma' in raw_df.columns else []
    )

    selected_odasayi = st.sidebar.selectbox(
        "Oda Sayısı Seçiniz",
        sorted(raw_df['odasayi'].dropna().unique(), key=lambda x: str(x)) if 'odasayi' in raw_df.columns else []
    )

    selected_esya = st.sidebar.selectbox(
        "Eşya Durumu Seçiniz",
        sorted(raw_df['esya'].dropna().unique(), key=lambda x: str(x)) if 'esya' in raw_df.columns else []
    )

    selected_balkon = st.sidebar.selectbox(
        "Balkon Durumu Seçiniz",
        sorted(raw_df['balkon'].dropna().unique(), key=lambda x: str(x)) if 'balkon' in raw_df.columns else []
    )

    if selected_balkon.lower() == "var":
        try:
            raw_df['balkonsayi'] = pd.to_numeric(raw_df['balkonsayi'], errors='coerce')
            min_balkon = int(raw_df['balkonsayi'].min())
            max_balkon = int(raw_df['balkonsayi'].max())
        except Exception:
            min_balkon, max_balkon = 0, 10
        selected_balkonsayi = st.sidebar.number_input("Balkon Sayısı Giriniz", min_value=min_balkon, max_value=max_balkon, value=1, step=1)
    else:
        selected_balkonsayi = 0

    numeric_fields = ['metrekare', 'binayas', 'binakat', 'banyosayi', 'dairekat']
    num_values = {}
    for col in numeric_fields:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
            min_val = int(raw_df[col].min())
            max_val = int(raw_df[col].max())
            mean_val = int(raw_df[col].mean())
            num_values[col] = (min_val, max_val, mean_val)

    selected_metrekare = st.sidebar.number_input("Metrekare Giriniz", min_value=num_values['metrekare'][0], max_value=num_values['metrekare'][1], value=num_values['metrekare'][2], step=1)
    selected_binayas = st.sidebar.number_input("Bina Yaşı Giriniz", min_value=num_values['binayas'][0], max_value=num_values['binayas'][1], value=num_values['binayas'][2], step=1)
    selected_binakat = st.sidebar.number_input("Bina Katı Giriniz", min_value=num_values['binakat'][0], max_value=num_values['binakat'][1], value=num_values['binakat'][2], step=1)
    selected_dairekat = st.sidebar.number_input("Daire Katı Giriniz", min_value=num_values['dairekat'][0], max_value=num_values['dairekat'][1], value=num_values['dairekat'][2], step=1)

    # Banyosayi için ortalama değer alınıyor
    if 'banyosayi' in raw_df.columns:
        selected_banyosayi = int(raw_df['banyosayi'].dropna().mean())
    else:
        selected_banyosayi = 1

    model_option = st.sidebar.selectbox("Model Seçiniz", ["Karar Ağacı", "SVR", "Yapay Sinir Ağı"])

    model_file = 'trained_models.pkl'
    if not os.path.exists(model_file):
        st.error(f"{model_file} bulunamadı. Lütfen modeli train_and_save.py ile eğitip dosyayı oluşturun.")
        st.stop()

    try:
        saved_model = joblib.load(model_file)
        models = saved_model["models"]
        scores = saved_model["scores"]
        model_columns = saved_model["model_columns"]
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        st.stop()

    input_data = {
        "ilce": [selected_ilce],
        "mahalle": [selected_mahalle],
        "tip": [selected_tip],
        "isitma": [selected_isitma],
        "odasayi": [selected_odasayi],
        "esya": [selected_esya],
        "balkon": [selected_balkon],
        "metrekare": [selected_metrekare],
        "binayas": [selected_binayas],
        "binakat": [selected_binakat],
        "banyosayi": [selected_banyosayi],
        "dairekat": [selected_dairekat],
        "balkonsayi": [selected_balkonsayi]
    }

    input_df = pd.DataFrame(input_data)
    input_transformed = transform_new_data(input_df, model_columns)

    if st.button("Fiyat Tahmini Al"):
        model = models[model_option]
        prediction = model.predict(input_transformed)[0]
        st.success(f"Seçilen ({model_option}) modele göre tahmini konut fiyatı: {prediction:,.2f} TL")
        st.info(f"Model başarı oranı (R² skoru): {round(scores[model_option], 2)}")

if __name__ == '__main__':
    main()
