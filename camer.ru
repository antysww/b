from ultralystics import YOLO
import cv2
#загружаем модель yolo скелет человека 
#создфем переменную для хранение модели
model = YOLO('yolo8n-pose.pt')
#переменная для запуска камеры подключенной пк или встроенной ноута
cap = cv2.VideoCapture()

while True:
    ref,frame = cap.read()
    if not ret:
        break
        #детекция позы человека
        result = model(frame,verbose=False[0])

        #рисуем результат
        annotated_frame = result.plot()
        
