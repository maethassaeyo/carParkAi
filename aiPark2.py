import numpy as np
import cv2
import time
from flask import Flask, render_template, Response, jsonify
import logging
from datetime import datetime
app = Flask(__name__)


# รายชื่อหมวดหมู่ทั้งหมด เรียงตามลำดับ
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
           "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
           "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
           "SOFA", "TRAIN", "TVMONITOR"]
# สีตัวกรอบที่วาดrandomใหม่ทุกครั้ง
COLORS = np.random.uniform(0, 100, size=(len(CLASSES), 3))
# โหลดmodelจากแฟ้ม
net = cv2.dnn.readNetFromCaffe(
    "./MobileNetSSD/MobileNetSSD.prototxt",
    "./MobileNetSSD/MobileNetSSD.caffemodel"
)
# เลือกวิดีโอ/เปิดกล้อง
IP = "192.168.0.107"
URL = 'rtsp://admin:URGHJA@'+IP+':554/h264_stream'
#URL = './vdo/4.mp4'
numcar = 0

def write(message,level='INFO'):
    timestamp = datetime.now().strftime('%Y-%m%d %H:%M:%S')
    log_entry = f'[{timestamp}] {level}:{message}\n'
    
    with open('numcar.log','a',encoding='utf-8') as f:
        f.write(log_entry)


def open_rtsp_stream():
    print("🎥 Connecting to RTSP stream...")
    cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลด delay
    if not cap.isOpened():
        print("❌ Cannot open RTSP stream")
        return None
    print("✅ Connected!")
    return cap

cap = open_rtsp_stream()

fail_count = 0

def gen_frames():
    global numcar
    global fail_count
    global cap
    while True:
        
        ret, frame = cap.read()
        #frame = cv2.resize(frame,(1200,600))
        if not ret or frame is None:
            print("⚠️ Frame read failed — skipping...")
            fail_count += 1
            time.sleep(0.1)
            if fail_count > 10:
                print("🔁 Reconnecting to RTSP...")
                cap.release()
                cap = open_rtsp_stream()
                fail_count = 0
            continue

        fail_count = 0  # reset counter


        if ret:
            (h, w) = frame.shape[:2]
            # ทำpreprocessing
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            # feedเข้าmodelพร้อมได้ผลลัพธ์ทั้งหมดเก็บมาในตัวแปร detections
            detections = net.forward()
            numcar = 0
            for i in np.arange(0, detections.shape[2]):
                percent = detections[0, 0, i, 2]
       
                if percent > 0.6:
                    class_index = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                  
                    label = "{} [{:.2f}%]".format(
                        CLASSES[class_index], percent * 100)
                    numcar+=1 if CLASSES[class_index] == "CAR" else numcar
                    cv2.rectangle(frame, (startX, startY),
                                (endX, endY), COLORS[class_index], 2)
                    cv2.rectangle(frame, (startX - 1, startY - 30),
                                (endX + 1, startY), COLORS[class_index], cv2.FILLED)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX + 20, y + 5),cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            #cv2.putText(frame, "There ard {} car".format((str(numcar))), (10+ 20, 25 + 5),cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            #cv2.putText(frame, "available ard {} park".format((str(6-numcar))), (10+ 20, 50),cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.line(frame,(400,0),(400,1000),(0,0,255),2)
            cv2.line(frame,(800,0),(800,1000),(0,0,255),2)
            cv2.line(frame,(1200,0),(1200,1000),(0,0,255),2)
            cv2.line(frame,(1600,0),(1600,1000),(0,0,255),2)
            # encode
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/numcar')
def get_numcar():
    write(numcar)
    return jsonify({"numcar": numcar})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

