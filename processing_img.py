import cv2
import numpy as np
import os
from tqdm import tqdm

# Папки для входных и выходных данных
INPUT_FOLDER = r"C:\Users\Windows 10 Lite\PycharmProjects\Neural_network-Rusal\Datasets"  # Папка с исходными изображениями
OUTPUT_FOLDER = r"C:\Users\Windows 10 Lite\PycharmProjects\Neural_network-Rusal\Datasets_processing_train"  # Папка для сохранённых изображений
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def enhance_contrast(image):

    # Применение CLAHE для LAB-пространства
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))  # Увеличенный clipLimit
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Применение гамма-коррекции для усиления яркости
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)

    # Линейное растяжение гистограммы
    alpha = 1.5
    beta = 0
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    return enhanced

def crop_sides(image, crop_percent=0.3):

    h, w, _ = image.shape
    crop_w = int(w * crop_percent)
    cropped = image[:, crop_w:w - crop_w]
    return cropped

def process_first_stage(image_path):

    image = cv2.imread(image_path)
    if image is None:
        return None

    # Повышение контрастности
    enhanced = enhance_contrast(image)

    # Обрезка боковых сторон
    cropped = crop_sides(enhanced)

    return cropped

def enhance_contrast_second_stage(image):

    # Применение CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

def extract_edges(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Бинаризация с порогом Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Поиск контуров Canny
    edges = cv2.Canny(binary, 50, 150)

    # Увеличение контуров с помощью морфологии
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    return edges

def mask_large_objects(edges, min_area=500):

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask

def process_second_stage(image):

    # Повышение контрастности
    enhanced = enhance_contrast_second_stage(image)

    # Выделение контуров
    edges = extract_edges(enhanced)

    # Маскирование объектов
    mask = mask_large_objects(edges)

    # Наложение маски на изображение
    result = cv2.bitwise_and(enhanced, enhanced, mask=mask)


    return result

def process_image(image_path, output_folder):

    # Первый этап обработки
    first_stage_result = process_first_stage(image_path)
    if first_stage_result is None:
        return

    # Второй этап обработки
    final_result = process_second_stage(first_stage_result)

    # Сохранение итогового изображения
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{basename}_final_processed.jpg")
    cv2.imwrite(output_path, final_result)


# Применение обработки ко всем изображениям
for file_name in tqdm(os.listdir(INPUT_FOLDER)):
    input_path = os.path.join(INPUT_FOLDER, file_name)
    process_image(input_path, OUTPUT_FOLDER)


