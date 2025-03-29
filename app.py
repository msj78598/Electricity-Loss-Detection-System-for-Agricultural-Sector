import os
import math
import pandas as pd
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import joblib
from sklearn.preprocessing import StandardScaler
import streamlit as st
import urllib.parse

# إعدادات صفحة Streamlit
st.set_page_config(page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية", layout="wide")

# إعدادات API للقمر الصناعي
API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"
ZOOM = 15
IMG_SIZE = 640
MAP_TYPE = "satellite"

# إعداد المسارات النسبية للملفات
IMG_DIR = os.path.join(os.getcwd(), "images")
DETECTED_DIR = os.path.join(os.getcwd(), "DETECTED_FIELDS", "FIELDS", "farms")
MODEL_PATH = os.path.join(os.getcwd(), "best.pt")
ML_MODEL_PATH = os.path.join(os.getcwd(), "model", "final_model.joblib")
OUTPUT_EXCEL = os.path.join(os.getcwd(), "output", "detected_low_usage.xlsx")

# إعدادات المجلدات
Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(DETECTED_DIR).mkdir(parents=True, exist_ok=True)
Path("output").mkdir(parents=True, exist_ok=True)

# تحميل صورة القمر الصناعي
def download_image(lat, lon, meter_id):
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": f"{IMG_SIZE}x{IMG_SIZE}",
        "maptype": MAP_TYPE,
        "key": API_KEY
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(response.content)
            return img_path
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# تحويل بكسل لمساحة
def pixel_to_area(lat, box):
    scale = 156543.03392 * abs(math.cos(math.radians(lat))) / (2 ** ZOOM)
    width_m = abs(box[2] - box[0]) * scale
    height_m = abs(box[3] - box[1]) * scale
    return width_m * height_m

# تحليل صورة واحدة بـ YOLOv5
def detect_field(img_path, meter_id, info, model):
    results = model(img_path)
    df_result = results.pandas().xyxy[0]
    fields = df_result[df_result["name"] == "field"]
    if not fields.empty:
        confidence = round(fields["confidence"].max() * 100, 2)
        if confidence >= 85:
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            largest_field = fields.iloc[0]
            box = [largest_field["xmin"], largest_field["ymin"], largest_field["xmax"], largest_field["ymax"]]
            draw.rectangle(box, outline="green", width=3)
            area = pixel_to_area(info['y'], box)
            draw.text((10, 10), f"ID: {meter_id}\nArea: {int(area)} m²", fill="yellow")
            image_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
            image.save(image_path)
            return confidence, image_path, int(area)
    return None, None, None

# تحديد الأولوية
def determine_priority(has_field, anomaly, consumption_check, high_priority_condition):
    if high_priority_condition:
        return "أولوية عالية جدًا"
    elif has_field and anomaly == 1 and consumption_check:
        return "قصوى"
    elif has_field and (anomaly == 1 or consumption_check):
        return "متوسطة"
    elif has_field:
        return "منخفضة"
    return "طبيعية"

# تشغيل نموذج ML على حالة
def predict_loss(info, model_ml):
    X = [[info["Breaker Capacity"], info["الكمية"]]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return model_ml.predict(X_scaled)[0]

# زر مشاركة الحالة عبر WhatsApp
def generate_whatsapp_share_link(meter_id, confidence, area, location_link, quantity, capacity, office_number, priority):
    message = f"حالة عداد {meter_id}:\n" \
              f"رقم المكتب: {office_number}\n" \
              f"أولوية الحالة: {priority}\n" \
              f"ثقة: {confidence}%\n" \
              f"مساحة تقديرية: {area} م²\n" \
              f"كمية الاستهلاك: {quantity} كيلو\n" \
              f"سعة القاطع: {capacity} أمبير\n" \
              f"رابط الموقع: {location_link}"
    url = f"https://wa.me/?text={urllib.parse.quote(message)}"
    return url

# عرض موقع الحالة على Google Maps
def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

# واجهة المستخدم
st.title("🔍 نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية")

uploaded_file = st.file_uploader("📁 ارفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df["cont"] = df["الاشتراك"].astype(str).str.strip()
    df["المكتب"] = df["المكتب"].astype(str)
    df["الكمية"] = pd.to_numeric(df["الكمية"], errors="coerce")

    model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
    model_ml = joblib.load(ML_MODEL_PATH)

    st.success("✅ تم رفع الملف بنجاح")
    progress = st.progress(0)

    download_placeholder = st.empty()
    gallery_placeholder = st.empty()

    results = []
    for idx, row in df.iterrows():
        meter_id = str(row["cont"])
        lat, lon = row['y'], row['x']
        office_number = row["المكتب"]
        img_path = download_image(lat, lon, meter_id)
        if img_path:
            conf, img_detected, area = detect_field(img_path, meter_id, row, model_yolo)
            if conf:
                anomaly = predict_loss(row, model_ml)
                capacity_limit = capacity_thresholds.get(row['Breaker Capacity'], 0)
                consumption_check = row['الكمية'] < 0.5 * capacity_limit
                high_priority_condition = (conf >= 85 and row['الكمية'] == 0) or (conf >= 85 and row['Breaker Capacity'] < 200)
                priority = determine_priority(conf >= 85, anomaly, consumption_check, high_priority_condition)

                row["نسبة_الثقة"] = conf
                row["الأولوية"] = priority
                row["المساحة"] = area
                results.append(row)

                location_link = generate_google_maps_link(lat, lon)
                whatsapp_link = generate_whatsapp_share_link(meter_id, conf, area, location_link, row['الكمية'], row['Breaker Capacity'], office_number, priority)

                df_final = pd.DataFrame(results)
                with download_placeholder:
                    with open(OUTPUT_EXCEL, "wb") as f:
                        df_final.to_excel(f, index=False)
                    with open(OUTPUT_EXCEL, "rb") as f:
                        st.download_button("📥 تحميل النتائج", data=f, file_name="detected_low_usage.xlsx", key=f"download_button_{len(results)}")

                with gallery_placeholder.container():
                    st.image(img_detected, caption=f"عداد: {meter_id}\nثقة: {conf}%\nمساحة: {area} م²\n{priority}\nالمكتب: {office_number}\nالكمية: {row['الكمية']} كيلو واط", width=150)
                    st.markdown(f"🔗 [مشاركة]({whatsapp_link})")
                    st.markdown(f"📍 [Google Maps]({location_link})")

        progress.progress((idx + 1) / len(df))
