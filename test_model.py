import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def plot_predictions(results):
    # Извлекаем имена изображений и их предсказания
    images = list(results.keys())
    predictions = list(results.values())

    # Создаем график для предсказаний
    plt.figure(figsize=(12, 7))
    plt.title('График предсказаний')
    plt.plot(images, predictions, color='blue')
    plt.xticks(ticks=np.arange(0, len(images), step=5), labels=images[::5], rotation=45)
    plt.axhline(y=2.5, color='red', linestyle='--', label='y=2.5')
    plt.axhline(y=7.5, color='red', linestyle='--', label='y=7.5')
    plt.title('Предсказания модели')
    plt.xlabel('Изображение')
    plt.ylabel('Предсказание')
    plt.grid(True)
    plt.legend()
    plt.show()


def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        rating = output.item()
        return rating
def load_model(model_path, device):
    model = models.resnet152(weights=None)

    # Заменяем последний слой на линейный для регрессии
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 1)  # Один выход для регрессии
    )

    # Загружаем веса с частичной загрузкой
    checkpoint = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()

    # Оставляем только совпадающие слои
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Обновляем модель
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def test_model(model_path, test_images_dir, device):
    model = load_model(model_path, device)

    results = {}
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        if not os.path.isfile(image_path):
            continue

        image_tensor = prepare_image(image_path)
        rating = predict(model, image_tensor, device)
        results[image_name] = rating

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_images_dir = "Datasets_processing_test"  # Путь к тестовым изображениям
    # Прогнозирование и вывод предсказаний на тестовых изображениях
    results = test_model("ore_model_resnet152.pth", test_images_dir, device)

    for image_name, rating in results.items():
        print(f"{image_name}, Prediction: {rating:.2f}")

    plot_predictions(results)


main()


