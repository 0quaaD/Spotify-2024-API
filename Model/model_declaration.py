import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
import seaborn as sns

print(os.getcwd())

df = pd.read_csv('../Dataset/Most Streamed Spotify Songs 2024.csv', on_bad_lines='warn', encoding='latin1')

df = df.dropna(axis=0) # Drop the NaN values by columns

df['All Time Rank'] = df['All Time Rank'].str.replace(',','').astype(float)
df = df.drop(columns=['Track', 'Album Name', 
                      'Artist', 'Release Date', 
                      'ISRC', 'YouTube Views', 
                      'Soundcloud Streams','Explicit Track',
                      'Pandora Track Stations','Pandora Streams',
                      'All Time Rank'],axis=1)

X = df.drop(columns='Track Score',axis=1)
y = df['Track Score']
print(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2,random_state=42)

def gradient_model(train_X, train_y, test_y, test_X):
    model_gradient = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.001, max_depth=9, min_samples_leaf=4, min_samples_split=4, random_state=42)

    model_gradient.fit(train_X, train_y)

    y_pred_grad = model_gradient.predict(test_X)
    r2_ = r2_score(test_y, y_pred_grad)
    mse = mean_squared_error(test_y, y_pred_grad)
    print(f"MSE Gradient: {mse}")
    with open("grad_model.pkl",'wb') as file:
        pickle.dump(model_gradient,file)

    plt.figure(figsize=(10,6))
    plt.scatter(test_y, y_pred_grad, edgecolors='k',alpha=0.7)
    plt.plot([test_y.min(), test_y.max()], [y_pred_grad.min(), y_pred_grad.max()], color='r',linestyle='--')
    plt.xlabel("Actual Track Score")
    plt.ylabel("Predicted Track Score")
    plt.title("Gradient Boosting Model Prediction")
    plt.tight_layout()
    plt.show()
    return r2_

def xgb_model(train_X, train_y,test_X, test_y):
    model_xgb = XGBRegressor(n_estimators=10000, max_depth=10, eta=0.001, n_jobs=15, subsample=0.7, colsample_bytree=0.8)
    model_xgb.fit(train_X, train_y)
    y_pred_xgb = model_xgb.predict(test_X)
    r2_xgb = r2_score(test_y, y_pred_xgb)
    mse = mean_squared_error(test_y, y_pred_xgb)
    print(f"MSE XGB: {mse}")
    with open("xgb_model.pkl",'wb') as file:
        pickle.dump(model_xgb,file)
    
    plt.figure(figsize=(10,6))
    plt.scatter(test_y, y_pred_xgb, edgecolors='k',alpha=0.7)
    plt.plot([test_y.min(), test_y.max()], [y_pred_xgb.min(), y_pred_xgb.max()], color='r',linestyle='--')
    plt.xlabel("Actual Track Score")
    plt.ylabel("Predicted Track Score")
    plt.title("XGB Model Prediction")
    plt.tight_layout()
    plt.show()
    return r2_xgb

def forest_model(train_X, train_y, test_y, test_X):
    model_forest = RandomForestRegressor(n_estimators=500, criterion='squared_error',max_depth=10,n_jobs=12,random_state=42)
    model_forest.fit(train_X, train_y)
    y_pred_forest = model_forest.predict(test_X)
    r2_forest = r2_score(test_y, y_pred_forest)
    mse = mean_squared_error(test_y, y_pred_forest)
    print(f"MSE Forest: {mse:.4f}")
    with open("random_forest_model.pkl",'wb') as file:
        pickle.dump(model_forest,file)

    plt.figure(figsize=(10,6))
    plt.scatter(test_y, y_pred_forest, edgecolors='k',alpha=0.7)
    plt.plot([test_y.min(), test_y.max()], [y_pred_forest.min(), y_pred_forest.max()], color='r',linestyle='--')
    plt.xlabel("Actual Track Score")
    plt.ylabel("Predicted Track Score")
    plt.title("Random Forest Model Prediction")
    plt.tight_layout()
    plt.show()
    return r2_forest

# ✅ Corrected R² Metric
@tf.keras.utils.register_keras_serializable(package='Custom')
def r2_metrics(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_total + tf.keras.backend.epsilon())

# ✅ Improved TensorFlow model function
def tf_model(X, y):
    # ✅ Split data
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Normalize data
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    # ✅ Define model
    model_df = tf.keras.Sequential([
        tf.keras.Input(shape=(train_X.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    # ✅ Compile model with learning rate schedule
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
    )

    model_df.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[r2_metrics]
    )

    # ✅ Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    # ✅ Train model
    model_df.fit(
        train_X, train_y,
        validation_split=0.2,
        epochs=500,
        batch_size=64,
        verbose=1,
        callbacks=[early_stop]
    )

    # ✅ Evaluate model
    loss, r2_eval = model_df.evaluate(test_X, test_y)
    print(f"Test R^2 Score TensorFlow: {r2_eval:.4f}")

    # ✅ Predict and calculate external metrics
    y_pred_tf = model_df.predict(test_X).flatten()
    r2_res = r2_score(test_y, y_pred_tf)
    mse = mean_squared_error(test_y, y_pred_tf)
    print(f"MSE TensorFlow NN: {mse:.4f}")
    # ✅ Save model
    model_df.save("tensor_model.keras")


    plt.figure(figsize=(10,6))
    plt.scatter(test_y, y_pred_tf, edgecolors='k',alpha=0.7)
    plt.plot([test_y.min(), test_y.max()], [y_pred_tf.min(), y_pred_tf.max()], color='r',linestyle='--')
    plt.xlabel("Actual Track Score")
    plt.ylabel("Predicted Track Score")
    plt.title("TensorFlow NN Model Prediction")
    plt.tight_layout()
    plt.show()
    return r2_res

tf_ = tf_model(X,y) 
grad_ = gradient_model(train_X, train_y,test_y, test_X)
xgb_ = xgb_model(train_X, train_y,test_X, test_y)
forest_ = forest_model(train_X, train_y,test_y,test_X)

print(f'\nR^2 score Gradient Boosting: {grad_:.4f}\n')
print(f"R^2 Score TensorFlow: {tf_:.4f}\n")
print(f'R^2 Score XGBRegressor: {xgb_:.4f}\n')
print(f"R^2 Score Random Forest: {forest_:.4f}\n")


with open('grad_model.pkl', 'rb') as file:
    grad_model = pickle.load(file)

tensor_model = load_model("tensor_model.keras",custom_objects={'r2_metrics':r2_metrics})

with open('random_forest_model.pkl', 'rb') as file:
    forest_model = pickle.load(file)

with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

df.to_csv("../Dataset/Cleaned Data.csv", index=False)
def feat_imp(df, model_name, model_label):
    features = df.drop(columns='Track Score', axis=1).columns.tolist()
    imp = model_name.feature_importances_
    imp_df = pd.DataFrame({'Features':features, 'Importance':imp}).sort_values(by='Importance',ascending=True)
    
    plt.figure(figsize=(10,6))
    plt.barh(range(len(imp_df)), imp_df['Importance'], align='center')
    plt.yticks(range(len(imp_df)), imp_df['Features'])
    plt.xlabel(f"Importances for {model_label} Model")
    plt.tight_layout()
    plt.show()

def permutation_imp(model, X, y, metric=r2_score):
    baseline = metric(y, model.predict(X))
    importances = []

    for i in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])
        score = metric(y, model.predict(X_permuted))
        importances.append(baseline - score)

    return np.array(importances)

def tensor_feat_imp(model, model_name, df):
    imp = permutation_imp(tensor_model, test_X, test_y)
    
    features = df.drop(columns='Track Score', axis=1).columns.tolist()
    imp_df = pd.DataFrame({'Features': features, 'Importance': imp}).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(imp_df)), imp_df['Importance'], align='center')
    plt.yticks(range(len(imp_df)), imp_df['Features'])
    plt.xlabel(f"Permutation Importance ({model_name})")
    plt.tight_layout()
    plt.show()

feat_imp(df, grad_model, 'Gradient Boosting Regression')
feat_imp(df, forest_model, 'Random Forest Regression')
feat_imp(df, xgb_model, 'XGB Regression')
tensor_feat_imp(tensor_model, "TensorFlow Keras", df)

def grad_plot(X, y, y_pred):
    plt.figure(figsize=(10,6))
    plt.scatter(y, y_pred)
    plt.show()
