# CSA Monitor Project

## Описание
Проект **CSA Monitor** предназначен для мониторинга и анализа рисков в CSA (Cloud Security Alliance). Он использует машинное обучение для оценки и предсказания рисков на основе данных из CSV файла.

## Содержимое проекта
Проект включает следующие файлы:

- **app.py**: Основной скрипт, который выполняет загрузку данных, обработку и предсказание рисков.
- **app copy.py**: Копия основного скрипта, возможно с изменениями или для тестирования.
- **CSA_monitorv5_processed.csv**: Обработанный CSV файл, содержащий данные для анализа и обучения модели.
- **model_csa_risk.joblib**: Сохраненная модель машинного обучения, обученная для предсказания рисков.
- **model_csa_riskv1.joblib**: Вторая версия сохраненной модели машинного обучения.
- **rкомпиляция модели и обучение.ipynb**: Jupyter Notebook, содержащий код для компиляции и обучения модели машинного обучения.

## Требования
Для запуска проекта вам понадобятся следующие зависимости:
- Python 3.x
- pandas
- scikit-learn
- joblib
- Jupyter Notebook (если вы хотите запускать и редактировать .ipynb файл)

## Установка
1. Склонируйте репозиторий:
    ```bash
    git clone https://github.com/yourusername/CSA_Monitor.git
    cd CSA_Monitor
    ```
2. Установите необходимые зависимости:
    ```bash
    pip install -r requirements.txt
    ```

## Использование
1. Запустите `app.py` для выполнения анализа и предсказания рисков:
    ```bash
    python app.py
    ```
2. Вы можете изменять и запускать `rкомпиляция модели и обучение.ipynb` для повторного обучения модели или анализа данных.

# Автор Kizimov S. S. (nimblesheeld)

# Лицензия на использование программного обеспечения

Данное программное обеспечение, включая его исходный код, дизайн и алгоритмы, является собственностью [Kizimov S. S. (nimblesheeld)] и предоставляется пользователям на условиях проприетарной лицензии.

Пользователи имеют право использовать программное обеспечение только для личных  целей в его текущем виде. Любая модификация, декомпиляция, реверс-инжиниринг, а также распространение модифицированных версий программного обеспечения строго запрещены без предварительного письменного согласия [Kizimov S. S. (nimblesheeld)].

Любое использование программного обеспечения должно сопровождаться соблюдением всех условий, изложенных в настоящей лицензии. Нарушение условий лицензии может повлечь за собой юридическую ответственность.

Все права защищены. © [Kizimov S. S. (nimblesheeld) ], [2024].