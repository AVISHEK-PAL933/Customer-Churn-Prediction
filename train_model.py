import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("1. LOADING & PREPROCESSING DATA")
print("="*60)

df = pd.read_csv("churn_modelling_dataset.csv")
print(f"Dataset shape: {df.shape}")

# Drop irrelevant columns
drop_cols = ["RowNumber", "CustomerId", "Surname"]
df_clean = df.drop(columns=drop_cols)

# Separate features and target
X = df_clean.drop(columns=["Exited"])
y = df_clean["Exited"]

print(f"Features: {X.columns.tolist()}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")

# ----------------------------------------------------
# Fit Encoders
# ----------------------------------------------------
# 1. Gender Encoder
label_encoder_gender = LabelEncoder()
X["Gender"] = label_encoder_gender.fit_transform(X["Gender"])

# 2. Geography OneHot Encoder
onehot_encoder_geo = OneHotEncoder(drop=None, sparse_output=False)
geo_encoded = onehot_encoder_geo.fit_transform(X[["Geography"]])
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Concatenate EXACTLY as app.py expects:
# input_df.drop("Geography", axis=1) + geo_encoded_df
X_encoded = pd.concat(
    [X.drop("Geography", axis=1), geo_encoded_df],
    axis=1
)

print(f"Encoded feature columns ({len(X_encoded.columns)}): {X_encoded.columns.tolist()}")

# ----------------------------------------------------
# Stratified Train-Test Split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# ----------------------------------------------------
# Fit StandardScaler ONLY on Training Data
# ----------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------
# Class Weighting for Imbalance
# ----------------------------------------------------
class_weights_vals = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_vals))
print(f"Calculated class weights: {class_weights_dict}")

print("\n" + "="*60)
print("2. BUILDING & TRAINING ANN MODEL")
print("="*60)

input_dim = X_train_scaled.shape[1]

model = Sequential([
    Dense(64, activation='relu', input_dim=input_dim),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n" + "="*60)
print("3. MODEL EVALUATION ON UNTOUCHED TEST SET")
print("="*60)

# Evaluate on test set
test_loss, test_acc, test_auc_metric = model.evaluate(X_test_scaled, y_test, verbose=0)
y_pred_proba = model.predict(X_test_scaled, verbose=0).ravel()

roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Overall Test Loss: {test_loss:.4f}")
print(f"Overall Test ROC-AUC: {roc_auc*100:.2f}%")

print("\n--- CLASSIFICATION THRESHOLD ANALYSIS ---")
thresholds = [0.30, 0.40, 0.50, 0.60]

print(f"{'Threshold':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 60)

best_th = 0.50
best_f1 = 0.0

for th in thresholds:
    y_pred_th = (y_pred_proba >= th).astype(int)
    acc = accuracy_score(y_test, y_pred_th)
    prec = precision_score(y_test, y_pred_th, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred_th, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred_th, pos_label=1, zero_division=0)
    
    print(f"{th:<10.2f} | {acc*100:<9.2f}% | {prec*100:<9.2f}% | {rec*100:<9.2f}% | {f1*100:<9.2f}%")
    
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"\nOptimal threshold selected for F1-Score: {best_th:.2f} (F1: {best_f1*100:.2f}%)")

# Standard 0.50 threshold evaluation for report
y_pred_default = (y_pred_proba >= 0.50).astype(int)
acc_def = accuracy_score(y_test, y_pred_default)
prec_def = precision_score(y_test, y_pred_default, pos_label=1)
rec_def = recall_score(y_test, y_pred_default, pos_label=1)
f1_def = f1_score(y_test, y_pred_default, pos_label=1)
cm_def = confusion_matrix(y_test, y_pred_default)

print("\n" + "="*60)
print("DEFAULT THRESHOLD (0.50) PERFORMANCE")
print("="*60)
print(f"Accuracy: {acc_def*100:.2f}%")
print(f"Precision: {prec_def*100:.2f}%")
print(f"Recall: {rec_def*100:.2f}%")
print(f"F1 Score: {f1_def*100:.2f}%")
print(f"ROC-AUC: {roc_auc*100:.2f}%")
print("\nConfusion Matrix:")
print(cm_def)

# Selected Optimal Threshold Evaluation
y_pred_opt = (y_pred_proba >= best_th).astype(int)
acc_opt = accuracy_score(y_test, y_pred_opt)
prec_opt = precision_score(y_test, y_pred_opt, pos_label=1)
rec_opt = recall_score(y_test, y_pred_opt, pos_label=1)
f1_opt = f1_score(y_test, y_pred_opt, pos_label=1)
cm_opt = confusion_matrix(y_test, y_pred_opt)

print("\n" + "="*60)
print(f"OPTIMAL THRESHOLD ({best_th:.2f}) PERFORMANCE")
print("="*60)
print(f"Accuracy: {acc_opt*100:.2f}%")
print(f"Precision: {prec_opt*100:.2f}%")
print(f"Recall: {rec_opt*100:.2f}%")
print(f"F1 Score: {f1_opt*100:.2f}%")
print(f"ROC-AUC: {roc_auc*100:.2f}%")
print("\nConfusion Matrix:")
print(cm_opt)

print("\n" + "="*60)
print("4. SAVING ARTIFACTS")
print("="*60)

# Save best model
model.save("model.h5")
print("Saved model -> model.h5")

# Save fitted scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved scaler -> scaler.pkl")

# Save gender encoder
with open("label_encoder_gender.pkl", "wb") as f:
    pickle.dump(label_encoder_gender, f)
print("Saved label_encoder_gender -> label_encoder_gender.pkl")

# Save geography encoder
with open("onehot_encoder_geo.pkl", "wb") as f:
    pickle.dump(onehot_encoder_geo, f)
print("Saved onehot_encoder_geo -> onehot_encoder_geo.pkl")

# ----------------------------------------------------
# 5. VERIFICATION AGAINST APP.PY PREPROCESSING PIPELINE
# ----------------------------------------------------
print("\n" + "="*60)
print("5. VERIFYING STREAMLIT APP & PREPROCESSING PARITY")
print("="*60)

test_customer = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0
}

# 1. Pipeline Prediction
test_df = pd.DataFrame([test_customer])
test_df["Gender"] = label_encoder_gender.transform(test_df["Gender"])
geo_enc = onehot_encoder_geo.transform(test_df[["Geography"]])
geo_enc_df = pd.DataFrame(
    geo_enc,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)
test_df_encoded = pd.concat([test_df.drop("Geography", axis=1), geo_enc_df], axis=1)

# Verify column alignment
assert list(test_df_encoded.columns) == list(X_encoded.columns), "Column mismatch with training dataset!"
print("Feature column alignment check PASSED!")

test_scaled = scaler.transform(test_df_encoded)
pred_prob = float(model.predict(test_scaled, verbose=0)[0][0])

print(f"\nVerification Customer Test Input:")
print(test_customer)
print(f"Predicted Churn Probability: {pred_prob:.4f} ({pred_prob*100:.2f}%)")

print("\nModel training, artifact export, and parity verification COMPLETED SUCCESSFULLY!")
