import joblib
from utils import load_data, preprocess_data
from models import train_models

def main():
    print("Veri yükleniyor...")
    raw_df = load_data('HouseData2.xlsx')
    if raw_df is None:
        print("Veri yüklenirken hata oluştu.")
        return

    print("Veri ön işleniyor...")
    df_processed, model_columns = preprocess_data(raw_df.copy())

    print("Modeller eğitiliyor...")
    models, scores = train_models(df_processed)

    saved_model = {
        "models": models,
        "scores": scores,
        "model_columns": model_columns
    }

    print("Model dosyası kaydediliyor...")
    joblib.dump(saved_model, 'trained_models.pkl')
    print("Model başarıyla kaydedildi. 'trained_models.pkl' dosyası oluşturuldu.")

if __name__ == "__main__":
    main()
