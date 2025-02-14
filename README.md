# Нейросеть для Определения Размера Фракций Щебня

Этот проект реализует нейросеть для определения размера фракций щебня по изображениям. Он включает модули для предварительной обработки изображений, обучения модели и оценки её производительности.


## 📦 Структура Проекта
Последовательность выполнения скриптов для прохождения полного цикла работы нейросети:

1) **processing_img.py:** Выполняет предварительную обработку изображений.
2) **train_model.py:** Основной скрипт для обучения модели. Загружает данные, инициализирует нейросеть, выполняет обучение с мониторингом метрик (например, функция потерь MSE и R²).
3) **test_model.py:** Используется для тестирования обученной модели на новых данных. Загружает модель и датасет, оценивает точность и визуализирует результаты.


## 🚀 Возможности

- Мощная автоматическая обработка изображений с использованием методов компьютерного зрения.
- Гибкая архитектура модели с возможностью настройки параметров обучения.
- Мощные методы улучшения эффективности модели.
- Поддержка пользовательских датасетов.
- Визуализация результатов для анализа производительности.

## ⚙️ Требования

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- tqdm

Установка зависимостей:

```bash
pip install torch torchvision opencv-python numpy matplotlib scikit-learn tqdm
```

## 🖼️ Предварительная Обработка Изображений

Запуск кода:

```bash
python processing_img.py
```
Основные компоненты:
- **Обрезка ненужной информации**
- **Повышение контрастности**
- **Применение гамма-коррекции для увеличения яроксти**
- **Применение линейного растяжения гистограммы для увеличения контрастности**
- **Дополнительное повышение контрастности**
- **Выделение контуров**
- **Наложение масок на контуры, похожие на фракции руды щебня**
- **Бинаризация**
- **Морфологические операции**

## 📊 Обучение Модели

Запуск кода:

```bash
python train_model.py
```
Основные компоненты:
- Подгрузка данных для датасетов и их трансформация.
- создание и аугментация датасетов.
- Загрузка архитектуры модели с добавлением слоев для регрессии.
- Обучение и валидация модели с использованием всех необходимых деталей оптимизации.
- Построение графиков функции потерь "MSE"
 и точности коэффициента детерминации.

## 🧪 Тестирование Модели

Запуск кода:

```bash
python test_model.py
```
Основные компоненты:
- **Подгрузка данных для датасетов и и их трансформация.**
- **Загрузка обученной модели и применение её к тестовым данным.**
- **Вывод предсказаний и их визуализация на графике.**

## 📈 Метрики Производительности

- **Коэффициент детерминации (R²):** Оценивает, насколько хорошо модель объясняет вариацию в данных.
- **Функция потерь (MSE):** Показывает, насколько сильно модель делает ошибки.
- **Графики потерь:** Помогают отслеживать процесс обучения и выявлять переобучение.

## 📁 Организация Данных
Относительные пути от корневой папки проекта к данным для нейросети:

- **`Datasets/`:** Исходные изображения.
- **`Datasets_processing_train/`:** Обработанные изображения для обучения.
- **`labels.json`:** Файл с метками для регрессионного анализа.

## 📊 Пример Использования

```python
from PIL import Image
from utils import enhance_contrast

# Загрузка и обработка изображения
image = Image.open('example.jpg')
process_image(image)

# Предсказание размера фракций
from train_model import load_model
model = load_model('model.pth')
prediction = model(processed_image)
print(f'Размер фракций: {prediction.item()}')
```

## 📄 Лицензия

Этот проект лицензирован по лицензии MIT.

## 🙌 Благодарности

- Разработано с использованием PyTorch и OpenCV.
- Вдохновлено реальными промышленными задачами автоматизации производственных процессов.

