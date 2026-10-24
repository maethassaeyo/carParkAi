# 🚗 carParkAi - ระบบตรวจจับพื้นที่จอดรถอัจฉริยะ

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-available-blue)](https://www.docker.com/)

**carParkAi** เป็นโปรเจกต์ที่ใช้ Image Processing และ Machine Learning ในการตรวจจับว่าพื้นที่จอดรถในลานจอดรถนั้น "ว่าง" หรือ "ไม่ว่าง" แบบ Real-time ผ่านวิดีโอหรือกล้องวงจรปิด เพื่อช่วยอำนวยความสะดวกในการจัดการที่จอดรถ

---

## ✨ ฟีเจอร์หลัก (Features)
- ✅ **Real-time Detection:** ตรวจจับสถานะที่จอดรถได้ทันทีจากไฟล์วิดีโอหรือกล้อง Live Stream
- ✅ **Parking Slot Marking:** แบ่งพื้นที่จอดเป็น 5 ช่อง (P1 - P5) ตามตำแหน่งที่กำหนด
- ✅ **Status Counting:** แสดงจำนวนที่ว่างและที่จอดทั้งหมดบนหน้าจอ
- ✅ **Visual Indicators:** แสดงกรอบสีเมื่อตรวจพบรถ และสถานะสีเขียว/แดงบน Dashboard
- ✅ **Web Dashboard:** แสดงผลผ่านหน้าเว็บ Flask แบบ Real-time

---

## 🛠 เทคโนโลยีที่ใช้ (Tech Stack)
- **Language:** Python
- **Libraries:**
  - `OpenCV` (สำหรับประมวลผลภาพและโหลดโมเดล MobileNetSSD)
  - `NumPy` (สำหรับการคำนวณ Matrix)
  - `Flask` (สำหรับ Web Server และ Dashboard)
- **Model:** MobileNetSSD (Caffe)

---

## 🚀 การติดตั้งและใช้งาน (Docker)

### 1. เตรียมโปรเจกต์
```bash
git clone https://github.com/maethassaeyo/carParkAi.git
cd carParkAi
```

### 2. สร้าง Docker Image
```bash
docker build -t carparkai .
```

### 3. รัน Container
```bash
docker run -d -p 5000:5000 --name carparkai carparkai
```

### 4. เปิดใช้งาน
เปิดเบราว์เซอร์ไปที่:
```
http://localhost:5000
```

### คำสั่งที่ใช้งานบ่อย
```bash
# ดู log
docker logs -f carparkai

# หยุดการทำงาน
docker stop carparkai

# ลบ container
docker rm carparkai

# สร้าง image ใหม่หลังแก้โค้ด
docker build -t carparkai . && docker run -d -p 5000:5000 --name carparkai carparkai
```

> **หมายเหตุ:** หากต้องการใช้กล้อง RTSP ให้แก้ไขค่า `URL` ในไฟล์ `aiPark2.py` ก่อน build image

---

## 🔌 API Endpoints

| Method | Endpoint | คำอธิบาย |
|--------|----------|----------|
| GET | `/` | หน้า Web Dashboard |
| GET | `/video` | สตรีมวิดีโอ (MJPEG) |
| GET | `/numcar` | ข้อมูลสถานะช่องจอดแบบ JSON |

### ตัวอย่าง Response จาก `/numcar`
```json
{
  "numcar": 2,
  "P1": 1,
  "P2": 0,
  "P3": 1,
  "P4": 0,
  "P5": 0
}
```

- `numcar` — จำนวนรถที่ตรวจพบ
- `P1` - `P5` — สถานะช่องจอด (`1` = ไม่ว่าง, `0` = ว่าง)

---

## 📁 โครงสร้างโปรเจกต์

```
carParkAi/
├── aiPark2.py            # แอปหลัก (Flask + detection)
├── demo.py               # เวอร์ชันเดโม
├── test.py               # ทดสอบผ่านหน้าต่าง OpenCV
├── scan_network.py       # สแกนหา IP ในวง LAN
├── Dockerfile            # สำหรับรันผ่าน Docker
├── MobileNetSSD/         # โมเดล MobileNetSSD
├── templates/index.html  # Web Dashboard
├── vdo/                  # ไฟล์วิดีโอตัวอย่าง
└── numcar.log            # ไฟล์ log ที่ถูกสร้างอัตโนมัติ
```

---

## 📄 License

โปรเจกต์นี้จัดทำเพื่อการศึกษา สามารถนำไปพัฒนาต่อได้
