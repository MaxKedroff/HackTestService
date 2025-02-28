import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Загрузка всех моделей
models = {
    "numbers": YOLO("numbers.pt"),
    "wheels": YOLO("wheels.pt"),
    "wagon": YOLO("wagon.pt"),
    "damage": YOLO("damage.pt"),
    "collectors": YOLO("collectors.pt")
}

# Названия классов
class_names = {
    "wheels": {0: "wheel pair", 1: "wheel pair", 2: "wheel pair", 3: "wheel pair", 4: "wheel pair"},
    "numbers": {0: "number"},
    "wagon": {
        0: "wagon", 1: "wagon", 2: "wagon",
        3: "autorack", 4: "boxcar", 5: "cargo",
        6: "container", 7: "flatcar", 8: "flatcar",
        9: "gondola", 10: "hopper", 11: "locomotive",
        12: "passenger", 13: "tank"
    },
    "damage": {0: "damaged wagon"},
    "collectors": {0: "lower position", 1: "upper position"}
}

# Функция для обработки кадра
def process_frame(frame):
    for model_name, model in models.items():
        results = model.predict(source=frame, conf=0.6)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if model_name == "numbers" else \
                         (255, 0, 0) if model_name == "wheels" else \
                         (0, 0, 255) if model_name == "wagon" else \
                         (255, 255, 0) if model_name == "damage" else \
                         (0, 255, 255)  # collectors
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                class_id = int(result.boxes.cls[0])
                class_name = class_names[model_name].get(class_id, "unknown")
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

# Генератор видеопотока
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Маршрут для главной страницы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для видеопотока
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
