
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# モデルとスケーラーの読み込み
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

# 特徴量名（15個）に合わせた入力フォームを作成
st.title("4個以上の採卵が期待できるか予測")

st.markdown("### 必要な情報を入力してください：")

# 入力用UI（数値）
inputs = {}
feature_names = [
    "年齢", "BMI", "経妊数", "経産数", "内膜症因子",
    "卵管因子", "男性因子", "PCOS", "チョコ手術既往", "AMH値",
    "AMH測定後経過期間（月）", "AFC", "卵胞期LH値", "卵胞期FSH値", "卵胞期E2値"
]

for name in feature_names:
    inputs[name] = st.number_input(f"{name}", value=0.0, key=name)

# 予測ボタン
if st.button("予測する"):
    # 入力データをDataFrame → スケーリング
    input_array = np.array([list(inputs.values())])
    input_scaled = scaler.transform(input_array)

    # 予測
    pred_prob = model.predict(input_scaled)[0][0]
    pred_class = int(pred_prob > 0.5)

    # 結果表示
    st.markdown(f"### 予測確率: **{pred_prob:.2f}**")
    st.markdown(f"### 判定結果: **{'成功' if pred_class == 1 else '失敗'}**")
