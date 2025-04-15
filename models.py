from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import numpy as np

def train_models(df):
    X = df.drop('fiyat', axis=1)
    y = df['fiyat']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {}
    scores = {}

    # Karar Ağacı
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    scores['Karar Ağacı'] = r2_score(y_test, y_pred_dt)
    models['Karar Ağacı'] = dt

    # SVR (parametreler optimize edilebilir)
    svr = SVR(C=1.0, epsilon=0.1, kernel='rbf')
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    scores['SVR'] = r2_score(y_test, y_pred_svr)
    models['SVR'] = svr

    # Yapay Sinir Ağı (MLPRegressor)
    best_ann_score = -np.inf
    best_ann_model = None
    for neurons in [40, 70, 100]:
        ann = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=500, random_state=42)
        ann.fit(X_train, y_train)
        y_pred_ann = ann.predict(X_test)
        score_ann = r2_score(y_test, y_pred_ann)
        if score_ann > best_ann_score:
            best_ann_score = score_ann
            best_ann_model = ann
    scores['Yapay Sinir Ağı'] = best_ann_score
    models['Yapay Sinir Ağı'] = best_ann_model

    return models, scores
