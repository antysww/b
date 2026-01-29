from ultralystics import YOLO
import cv2
#загружаем модель yolo скелет человека 
#создфем переменную для хранение модели
model = YOLO('yolo8n-pose.pt')
#переменная для запуска камеры подключенной пк или встроенной ноута
cap = cv2.VideoCapture(0)

while True:
    ref,frame = cap.read()
    if not ret:
        break
        #детекция позы человека
        result = model(frame,verbose=False)[0]

        #рисуем результат
        annotated_frame = result.plot()
        #создаем скелет поверх отрисовки кадра человека с захватом колизии
        cv2.addWeight(frame,0.3,annotated_frame,0.7,0,annotated_frame)
        #выведем некбольшую информацию на экран
        cv2.putText(annotated_frame,f"Общее количество лбдей: {len(result.keypoints)}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,255),2)
#показать на экране
cv2.imshow("Детектор", annotated_frame)
if cv2.waitKey(1) & 0xFF == 27:
    break
    #зфпуск
    cap.release()
  cv2.destroyAllWindows()