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
import sys

# -------------------------
# إعدادات عامة
# -------------------------
st.set_page_config(page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية", layout="wide")

# إعدادات API للقمر الصناعي
API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', 'API_KEY_HERE')  # استبدل بمفتاح API الخاص بك
ZOOM = 15
IMG_SIZE = 640
MAP_TYPE = "satellite"

# إعدادات المسارات
IMG_DIR = "images"
DETECTED_DIR = "DETECTED_FIELDS/FIELDS/farms"
MODEL_PATH = "yolov5/best.pt"
ML_MODEL_PATH = "final_model.joblib"
OUTPUT_EXCEL = "output/detected_low_usage.xlsx"

Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(DETECTED_DIR).mkdir(parents=True, exist_ok=True)
Path("output").mkdir(parents=True, exist_ok=True)

gallery = set()
results = []

# -------------------------
# تحميل YOLOv5 من المسار المحلي
# -------------------------
sys.path.append('./yolov5')
from models.experimental import attempt_load
from utils.general import non_max_suppression

try:
    model_yolo = attempt_load(MODEL_PATH, map_location=torch.device('cpu'))
    model_ml = joblib.load(ML_MODEL_PATH)
except Exception as e:
    st.error(f"خطأ في تحميل النماذج: {e}")

# -------------------------
# تحميل صورة القمر الصناعي
# -------------------------
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
        else:
            st.error(f"فشل تحميل الصورة للعداد {meter_id}")
    except Exception as e:
        st.error(f"خطأ في تحميل الصورة: {e}")
        return None

# -------------------------
# تحليل الصورة بـ YOLOv5
# -------------------------
def detect_field(img_path, meter_id, info):
    try:
        img = Image.open(img_path).convert('RGB')
        results = model_yolo([img_path], size=640)
        df_result = results.pandas().xyxy[0]
        fields = df_result[df_result["name"] == "field"]
        
        if not fields.empty:
            confidence = round(fields["confidence"].max() * 100, 2)
            if confidence >= 85:
                draw = ImageDraw.Draw(img)
                for _, row in fields.iterrows():
                    box = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
                    draw.rectangle(box, outline="green", width=3)
                
                detected_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
                img.save(detected_path)
                return confidence, detected_path
        return None, None
    except Exception as e:
        st.error(f"خطأ في تحليل الصورة للعداد {meter_id}: {e}")
        return None, None

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔎 نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية")

uploaded_file = st.file_uploader("📥 ارفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ تم رفع الملف بنجاح")
    except Exception as e:
        st.error(f"خطأ في قراءة ملف Excel: {e}")

    progress = st.progress(0)

    for idx, row in df.iterrows():
        try:
            meter_id = str(row["الاشتراك"].strip())
            lat, lon = row['y'], row['x']
            img_path = download_image(lat, lon, meter_id)

            if img_path:
                conf, img_detected = detect_field(img_path, meter_id, row)
                if conf and img_detected:
                    st.image(img_detected, caption=f"عداد: {meter_id}, ثقة: {conf}%")
                else:
                    st.warning(f"لم يتم اكتشاف حقل للعداد {meter_id}")
        except Exception as e:
            st.error(f"خطأ أثناء المعالجة للعداد {meter_id}: {e}")

        progress.progress((idx + 1) / len(df))

    st.success("🎉 التحليل اكتمل!")
