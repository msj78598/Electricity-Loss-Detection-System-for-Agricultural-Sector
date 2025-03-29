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

# -------------------------
# إعدادات عامة
# -------------------------
# تعيين إعدادات صفحة Streamlit مثل العنوان وتنسيق الصفحة.
st.set_page_config(page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية", layout="wide")

# إعدادات API للقمر الصناعي
API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"  # مفتاح API الخاص بـ Google Maps
ZOOM = 15  # مستوى التكبير للصورة القمرية
IMG_SIZE = 640  # حجم الصورة
MAP_TYPE = "satellite"  # نوع الخريطة (قمر صناعي)

# إعدادات المجلدات والمسارات
IMG_DIR = "images"  # مسار حفظ صور القمر الصناعي
DETECTED_DIR = "DETECTED_FIELDS/FIELDS/farms"  # مسار حفظ الصور بعد التحليل
MODEL_PATH = "yolov5/farms_project/field_detector/weights/best.pt"  # مسار نموذج YOLOv5 المدرب
ML_MODEL_PATH = "model/final_model.joblib"  # مسار نموذج تعلم الآلة المدرب
OUTPUT_EXCEL = "output/detected_low_usage.xlsx"  # مسار حفظ ملف النتائج بصيغة Excel

# وضع ملف Excel الذي سيقوم المستخدم بتحميله
st.subheader("📥 تحميل نموذج البيانات المطلوب تحليله")
template_file = "assets/fram.xlsx"  # مسار نموذج البيانات داخل مجلد assets
st.download_button("📥 تحميل نموذج البيانات", open(template_file, "rb"), file_name="fram.xlsx")

# تعريف الحدود القصوى لاستهلاك الطاقة بناءً على سعة القواطع
capacity_thresholds = {
    20: 6000, 50: 15000, 70: 21000, 100: 30000, 150: 45000,
    200: 60000, 300: 90000, 400: 120000, 500: 150000
}

# التأكد من وجود المجلدات الضرورية، وإنشاءها إذا لم تكن موجودة
Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(DETECTED_DIR).mkdir(parents=True, exist_ok=True)
Path("output").mkdir(parents=True, exist_ok=True)

# إعداد المتغيرات لحفظ الصور والنتائج
gallery = set()
results = []

# -------------------------
# تحميل صورة القمر الصناعي
# -------------------------
def download_image(lat, lon, meter_id):
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")  # تحديد مسار الصورة بناءً على رقم العداد
    if os.path.exists(img_path):
        return img_path  # إذا كانت الصورة موجودة مسبقًا، إعادتها مباشرة
    base_url = "https://maps.googleapis.com/maps/api/staticmap"  # رابط API لخرائط Google
    params = {
        "center": f"{lat},{lon}",  # تحديد الإحداثيات
        "zoom": ZOOM,  # تحديد مستوى التكبير
        "size": f"{IMG_SIZE}x{IMG_SIZE}",  # تحديد حجم الصورة
        "maptype": MAP_TYPE,  # نوع الخريطة (قمر صناعي)
        "key": API_KEY  # مفتاح API
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)  # إرسال طلب لتحميل الصورة
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(response.content)  # حفظ الصورة
            return img_path
    except Exception as e:
        print(f"Error downloading image: {e}")  # طباعة أي خطأ يحدث أثناء التحميل
        return None

# -------------------------
# تحويل بكسل لمساحة
# -------------------------
def pixel_to_area(lat, box):
    scale = 156543.03392 * abs(math.cos(math.radians(lat))) / (2 ** ZOOM)  # حساب نسبة تحويل البكسل للمتر
    width_m = abs(box[2] - box[0]) * scale  # حساب عرض الحقل بالمتر
    height_m = abs(box[3] - box[1]) * scale  # حساب طول الحقل بالمتر
    return width_m * height_m  # إرجاع المساحة بالمتر المربع

# -------------------------
# تحليل صورة واحدة بـ YOLOv5
# -------------------------
def detect_field(img_path, meter_id, info, model):
    results = model(img_path)  # استخدام نموذج YOLOv5 لاكتشاف الحقول
    df_result = results.pandas().xyxy[0]  # تحويل النتائج إلى DataFrame
    fields = df_result[df_result["name"] == "field"]  # استخراج الحقول من النتائج
    if not fields.empty:
        confidence = round(fields["confidence"].max() * 100, 2)  # حساب نسبة الثقة
        if confidence >= 85:  # إذا كانت الثقة أكبر من 85%
            image = Image.open(img_path).convert("RGB")  # فتح الصورة
            draw = ImageDraw.Draw(image)  # رسم مستطيل حول الحقل
            largest_field = fields.iloc[0]  # الحصول على أكبر حقل
            box = [largest_field["xmin"], largest_field["ymin"], largest_field["xmax"], largest_field["ymax"]]  # تحديد مستطيل الحقل
            draw.rectangle(box, outline="green", width=3)  # رسم المستطيل
            area = pixel_to_area(info['y'], box)  # حساب المساحة
            draw.text((10, 10), f"ID: {meter_id}\nArea: {int(area)} m²", fill="yellow")  # إضافة النص على الصورة
            image_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")  # حفظ الصورة المعدلة
            image.save(image_path)
            return confidence, image_path, int(area)  # إرجاع الثقة، الصورة، والمساحة
    return None, None, None

# -------------------------
# تحديد الأولوية
# -------------------------
def determine_priority(has_field, anomaly, consumption_check, high_priority_condition):
    if high_priority_condition:  # تحقق من الشروط ذات الأولوية العالية
        return "أولوية عالية جدًا"
    elif has_field and anomaly == 1 and consumption_check:
        return "قصوى"  # الأولوية القصوى إذا كان هناك حقل والانحراف والاستهلاك غير طبيعي
    elif has_field and (anomaly == 1 or consumption_check):
        return "متوسطة"  # الأولوية المتوسطة إذا كان هناك حقل أو الاستهلاك غير طبيعي
    elif has_field:
        return "منخفضة"  # الأولوية المنخفضة إذا كان هناك حقل والاستهلاك طبيعي
    return "طبيعية"  # إذا لم يتم اكتشاف أي شيء

# -------------------------
# تشغيل نموذج ML على حالة
# -------------------------
def predict_loss(info, model_ml):
    X = [[info["Breaker Capacity"], info["الكمية"]]]  # إعداد البيانات لاستخدام نموذج تعلم الآلة
    scaler = StandardScaler()  # مقياس القياس القياسي للبيانات
    X_scaled = scaler.fit_transform(X)  # تحويل البيانات
    return model_ml.predict(X_scaled)[0]  # التنبؤ بحالة الفاقد بناءً على البيانات

# -------------------------
# زر مشاركة الحالة عبر WhatsApp
# -------------------------
def generate_whatsapp_share_link(meter_id, confidence, area, location_link, quantity, capacity, office_number, priority):
    message = f"حالة عداد {meter_id}:\n" \
              f"رقم المكتب: {office_number}\n" \
              f"أولوية الحالة: {priority}\n" \
              f"ثقة: {confidence}%\n" \
              f"مساحة تقديرية: {area} م²\n" \
              f"كمية الاستهلاك: {quantity} كيلو\n" \
              f"سعة القاطع: {capacity} أمبير\n" \
              f"رابط الموقع: {location_link}"
    url = f"https://wa.me/?text={urllib.parse.quote(message)}"  # إنشاء رابط مشاركة عبر WhatsApp
    return url

# -------------------------
# عرض موقع الحالة على Google Maps
# -------------------------
def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"  # إنشاء رابط لموقع الحالة على خرائط Google

# -------------------------
# Streamlit
# -------------------------
# عرض واجهة المستخدم
st.title("🔍 نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية")

uploaded_file = st.file_uploader("📁 ارفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)  # تحميل البيانات من ملف Excel
    df["cont"] = df["الاشتراك"].astype(str).str.strip()  # تحويل العمود إلى سلسلة نصية
    df["المكتب"] = df["المكتب"].astype(str)  # التأكد من أن "المكتب" موجود كعمود نصي
    df["الكمية"] = pd.to_numeric(df["الكمية"], errors="coerce")  # تحويل الكمية إلى أرقام

    # تحميل النماذج المدربة
    model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
    model_ml = joblib.load(ML_MODEL_PATH)  # تحميل نموذج تعلم الآلة

    st.success("✅ تم رفع الملف بنجاح")
    progress = st.progress(0)  # شريط التقدم

    download_placeholder = st.empty()  # Placeholder لتحميل الملف
    gallery_placeholder = st.empty()  # Placeholder لعرض الصور

    for idx, row in df.iterrows():  # تحليل البيانات لكل سطر في الملف
        meter_id = str(row["cont"])  # الحصول على رقم العداد
        lat, lon = row['y'], row['x']  # إحداثيات العداد
        office_number = row["المكتب"]  # رقم المكتب
        img_path = download_image(lat, lon, meter_id)  # تحميل صورة القمر الصناعي
        if img_path:
            # تحليل الصورة باستخدام YOLOv5
            conf, img_detected, area = detect_field(img_path, meter_id, row, model_yolo)
            if conf and (conf, img_detected) not in gallery:
                anomaly = predict_loss(row, model_ml)  # التنبؤ بحالة الفاقد
                capacity_limit = capacity_thresholds.get(row['Breaker Capacity'], 0)  # تحديد الحد الأقصى للاستهلاك
                consumption_check = row['الكمية'] < 0.5 * capacity_limit  # التحقق من الاستهلاك

                # إضافة شرط الأولوية العالية
                high_priority_condition = (conf >= 85 and row['الكمية'] == 0) or (conf >= 85 and row['Breaker Capacity'] < 200)

                priority = determine_priority(conf >= 85, anomaly, consumption_check, high_priority_condition)  # تحديد الأولوية

                # إضافة البيانات إلى النتائج
                row["نسبة_الثقة"] = conf
                row["الأولوية"] = priority
                row["المساحة"] = area
                results.append(row)
                gallery.add((conf, priority, img_detected, meter_id, area, lat, lon, row['الكمية'], row['Breaker Capacity'], office_number))

                # توليد رابط المشاركة عبر WhatsApp
                location_link = generate_google_maps_link(lat, lon)
                whatsapp_link = generate_whatsapp_share_link(meter_id, conf, area, location_link, row['الكمية'], row['Breaker Capacity'], office_number, priority)

                # حفظ النتائج في ملف Excel
                df_final = pd.DataFrame(results)
                with download_placeholder:
                    with open(OUTPUT_EXCEL, "wb") as f:
                        df_final.to_excel(f, index=False)
                    with open(OUTPUT_EXCEL, "rb") as f:
                        st.download_button("📥 تحميل النتائج", data=f, file_name="detected_low_usage.xlsx", key=f"download_button_{len(results)}")

                # عرض الصور مع أزرار المشاركة
                with gallery_placeholder.container():
                    # عرض الحالات ذات الأولوية العالية أولًا
                    high_priority = [entry for entry in gallery if entry[1] == "أولوية عالية جدًا"]
                    other_priority = [entry for entry in gallery if entry[1] != "أولوية عالية جدًا"]

                    # عرض الحالات ذات الأولوية العالية
                    cols = st.columns(6)
                    for i, (conf, priority, img_path, meter_id, area, lat, lon, quantity, capacity, office_number) in enumerate(high_priority):
                        with cols[i % 6]:
                            st.image(img_path, caption=f"عداد: {meter_id}\nثقة: {conf}%\nمساحة: {area} م²\n{priority}\nالمكتب: {office_number}\nالكمية: {quantity} كيلو واط\nسعة القاطع: {capacity} أمبير", width=150)
                            st.markdown(f"🔗 [مشاركة]({whatsapp_link})")  # رابط المشاركة عبر WhatsApp
                            st.markdown(f"📍 [Google Maps]({location_link})")  # رابط الموقع على خرائط Google

                    # عرض باقي الحالات
                    for i, (conf, priority, img_path, meter_id, area, lat, lon, quantity, capacity, office_number) in enumerate(other_priority):
                        with cols[i % 6]:
                            st.image(img_path, caption=f"عداد: {meter_id}\nثقة: {conf}%\nمساحة: {area} م²\n{priority}\nالمكتب: {office_number}\nالكمية: {quantity} كيلو واط\nسعة القاطع: {capacity} أمبير", width=150)
                            st.markdown(f"🔗 [مشاركة]({whatsapp_link})")  # رابط المشاركة عبر WhatsApp
                            st.markdown(f"📍 [Google Maps]({location_link})")  # رابط الموقع على خرائط Google

        progress.progress((idx + 1) / len(df))  # تحديث شريط التقدم
