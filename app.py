import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка модели и токенизатора
model = load_model('model4.h5')
with open('tokenizer4.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Определяем категории
categories = ['Очень плохо', 'Плохо', 'Посредственно', 'Хорошо', 'Отлично']

# Максимальная длина последовательности
MAX_LEN = 200

# Функция для предсказания
def predict_rating(description, directors, actors):
    # Объединение текста
    text = f"{description} {directors} {actors}"

    # Токенизация и преобразование в последовательность
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)

    # Предсказание модели
    prediction = model.predict(padded_sequence)
    predicted_category = categories[np.argmax(prediction)]

    return predicted_category

# Интерфейс Streamlit
st.title("Оценка фильма на основе описания, режиссеров и актеров")

st.subheader("Введите данные фильма")

# Ввод данных
description = st.text_area("Описание фильма", placeholder="Введите описание фильма...")
directors = st.text_input("Режиссеры", placeholder="Введите имена режиссеров...")
actors = st.text_input("Актеры", placeholder="Введите имена актеров...")

# Кнопка для предсказания
if st.button("Предсказать"):
    if description.strip() and directors.strip() and actors.strip():
        result = predict_rating(description, directors, actors)
        st.success(f"Оценка фильма: **{result}**")
    else:
        st.error("Пожалуйста, заполните все поля!")
