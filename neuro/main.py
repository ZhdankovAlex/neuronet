import cv2
import numpy as np

import time

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
# конфигурация нейронной сети
config_path = "cfg/yolov3.cfg"
# файл весов сети YOLO (предварительно обученные веса)
weights_path = "weights/yolov3.weights"
# если у изображений высокое разрешение,
# требуется увеличить параметр font_scale
font_scale = 1
thickness = 1
# загрузка всех меток классов (объектов)
# используем датасет COCO
LABELS = open("data/coco.names").read().strip().split("\n")
# генерация цветов для каждого объекта
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# загрузка сети YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
# получим все имена слоев
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    # нужно нормализовать, масштабировать и изменить изображение,
    # чтобы оно подходило в качестве входных данных
    h, w = image.shape[:2]
    # нормализует значения пикселей в диапазоне от 0 до 1,
    # изменит размер изображения до (416х416) и изменит его форму
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    # устанавливает большой двоичный объект в качестве входа сети
    net.setInput(blob)
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    # измерим, сколько времени это заняло в секундах
    time_took = time.perf_counter() - start
    print("Time took:", time_took)
    boxes, confidences, class_ids = [], [], []

    # нужно перебрать выходные данные нейронной сети и отбросить любой объект,
    # уровень достоверности которого меньше, чем параметр CONFIDENCE

    # цикл для каждого слоя
    for output in layer_outputs:
        # цикл для каждого обнаруженного объекта
        for detection in output:
            # извлекаем идентификатор класса (метку) и
            # вероятность обнаруженного объекта
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # отбрасываем слабые прогнозы
            if confidence > CONFIDENCE:
                # YOLO на самом деле возвращает центр(x, y),
                # высоту и ширину контура
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # получаем верхний левый угол контура
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # обновляем списки
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # нейронка будет циклически прогонять все предсказания
    # и сохранять только те объекты, которые точнее всего

    # немаксимальное подавление для удаления дублирующей рамки объекта
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    # если у изображений высокое разрешение,
    # требуется увеличить параметр font_scale
    font_scale = 1
    thickness = 1

    # если обнаружили хоть один объект
    if len(idxs) > 0:
        # цикл по всем индексам
        for i in idxs.flatten():
            # получаем координаты контура
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # рисуем рамку и подписываем
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{LABELS[class_ids[i]]}: {confidences[i]:.2f}"
            # вычислите ширину и высоту текста,
            # чтобы нарисовать прозрачные поля в качестве фона текста
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # добавить непрозрачность
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            # теперь поместим вероятность
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    # показать изображение с метками
    cv2.imshow("image", image)
    if ord("q") == cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()
