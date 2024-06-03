import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

# Функция для добавления дополнительных признаков Probability_X
def add_probability_features(data):
    for i in range(1, 10):
        data[f'Probability_{i}'] = (data['Probability'] >= i).astype(int)
    return data

# Загрузка и использование модели для предсказаний
def load_model(model_path):
    model = load(model_path)
    return model

st.title('Анализ рисков')

# Загрузка файла данных пользователем
uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")
if uploaded_file is not None:
    # Чтение данных, убеждаемся что 'ID' читается как строка
    data = pd.read_csv(uploaded_file, dtype={'ID': str})
    data['ID'] = data['ID'].str.replace('.', '')  # Удаление точек, если они есть

    # Преобразование датасета
    data_processed = add_probability_features(data)

    # Загрузка модели
    model_path = 'model_csa_riskv1.joblib'
    model = load_model(model_path)

    # Предсказание
    features_to_drop = ['ID', 'target_value', 'Probability']  # Удаляем 'ID' перед предсказанием
    features = data_processed.drop(features_to_drop, axis=1, errors='ignore')
    predictions = model.predict(features)
    data_processed['prediction'] = predictions

    # Создаем новый DataFrame для отображения результата
    data_risk = data_processed[data_processed['prediction'] == 1]
    data_risk = data_risk[['ID', 'prediction']].rename(columns={'ID': 'Название угрозы'})
    
    # Добавляем столбец "Вероятность" обратно в data_risk для отображения
    data_risk['Вероятность'] = data_processed['Probability']

    # Отображаем только те записи, где есть предсказание риска
    st.write("Результаты анализа (только с высоким риском):")
    st.write(data_risk.set_index('Название угрозы'))  # Столбец "Название угрозы" как индекс

    # Анализ и визуализация данных (пример)
    plt.figure(figsize=(10, 6))
    plt.hist(data_processed['Probability'], bins=20, color='skyblue')
    plt.title('Распределение вероятностей')
    plt.xlabel('Вероятность')
    plt.ylabel('Количество')
    st.pyplot(plt)
