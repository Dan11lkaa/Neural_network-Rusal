import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torchvision import models

# Класс для работы с данными
class CustomDataset(Dataset):
    def __init__(self, data_dir, labels, transform=None):
        self.data_dir = data_dir
        self.labels = labels
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]  # Предположим, что изображения .jpg

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_name = os.path.basename(image_path)
        label = self.labels.get(image_name, 0)  # Получаем метку для изображения

        if self.transform:
            image = self.transform(image)

        return image, label


def create_labels_file(data_dir, labels_file):
    if not os.path.exists(labels_file):
        labels = {}
        for image_name in os.listdir(data_dir):
            if image_name.endswith('.jpg'):
                labels[image_name] = 0  # Пример: Заполняем метки нулями по умолчанию
        with open(labels_file, 'w') as f:
            json.dump(labels, f)
        print(f"Файл {labels_file} был создан и метки инициализированы нулями.")
    else:
        print(f"Файл {labels_file} уже существует.")


def load_labels(labels_file):
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    return labels


def prepare_data(data_dir, labels_file):
    labels = load_labels(labels_file)

    if all(label == 0 for label in labels.values()):
        print("Все метки равны 0, обучение невозможно.")
        return None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(data_dir, labels, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def build_model():
    # 1. Создаем модель ResNet152
    model = models.resnet152(weights=None)  # Здесь weights=None — без предобученных весов

    # 2. Заменяем последний слой (fc) на кастомный для задачи регрессии
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 1)
    )

    # 3. Возвращаем готовую модель
    return model



# График потерь и R2 на протяжении обучения и валидации
def plot_loss_and_mae(train_losses, val_losses, train_mae, val_mae, num_epochs):
    epochs = np.arange(1, num_epochs + 1)

    # График потерь
    plt.figure(figsize=(12, 5))
    plt.title('График функции потерь')
    plt.subplot(1, 2, 1)  # 1 строка, 2 колонки, первый график
    plt.plot(epochs, train_losses, label='Потери на обучении', color='blue')
    plt.plot(epochs, val_losses, label='Потери на валидации', color='skyblue', linestyle='--')
    plt.title('График Потерь(MSE)')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    # График MAE
    plt.subplot(1, 2, 2)  # 1 строка, 2 колонки, второй график
    plt.plot(epochs, train_mae, label='R2 на обучении', color='red')
    plt.plot(epochs, val_mae, label='R2 на валидации', color='orange', linestyle='--')
    plt.title('График R2')
    plt.xlabel('Эпоха')
    plt.ylabel('R2')
    plt.legend()
    plt.grid(True)

    # Показываем оба графика
    plt.tight_layout()
    plt.show()


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)
            val_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    mae = r2_score(all_labels, all_predictions)
    val_loss /= len(val_loader)
    return val_loss, mae


def train_model(model, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.MSELoss()  # Для регрессии используем MSE
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4, cooldown=2, min_lr=1e-6
    )

    num_epochs = 90
    train_losses = []
    val_losses = []
    train_mae = []
    val_mae = []

    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().detach().numpy())

        train_loss = running_loss / len(train_loader)
        train_mae_value = r2_score(all_labels, all_predictions)

        val_loss, val_mae_value = validate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mae.append(train_mae_value)
        val_mae.append(val_mae_value)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train MSE: {train_loss:.4f}, Train R2: {train_mae_value:.4f}, Val MSE: {val_loss:.4f}, Val R2: {val_mae_value:.4f}")

        # Проверка на улучшение
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), "ore_model_resnet152.pth")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Ранняя остановка, так как валидационная ошибка не улучшается.")
            break

    plot_loss_and_mae(train_losses, val_losses, train_mae, val_mae, num_epochs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_dir = "Datasets_processing_train"  # Путь к обучающим данным
    labels_file = "labels.json"  # Путь к файлу с метками

    # Создаем файл меток, если его нет
    create_labels_file(train_data_dir, labels_file)

    # Подготовка данных
    train_loader, val_loader = prepare_data(train_data_dir, labels_file)
    if not train_loader or not val_loader:
        return

    # Создание и обучение модели
    model = build_model()
    train_model(model, train_loader, val_loader, device)


main()
