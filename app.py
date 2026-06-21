import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def fix_keras_error():
    try:
        import keras.src.legacy.saving.serialization as legacy_serialization
        def strip_keys(d):
            if isinstance(d, dict):
                d.pop('quantization_config', None)
                d.pop('optional', None)
                for v in d.values(): strip_keys(v)
            elif isinstance(d, list):
                for item in d: strip_keys(item)
        orig = legacy_serialization.deserialize_keras_object
        def patched(identifier, *args, **kwargs):
            strip_keys(identifier)
            return orig(identifier, *args, **kwargs)
        legacy_serialization.deserialize_keras_object = patched
    except: pass
fix_keras_error()


st.set_page_config(page_title="Canteen AI Smart Checkout", page_icon="🍱", layout="wide")

BOX_COORDS = {
    "O_Canh": {"x": 0.16, "y": 0.10, "w": 0.33, "h": 0.49}, 
    "O_Com":  {"x": 0.56, "y": 0.12, "w": 0.25, "h": 0.47}, 
    "O_M1":   {"x": 0.18, "y": 0.61, "w": 0.20, "h": 0.31}, 
    "O_M2":   {"x": 0.39, "y": 0.62, "w": 0.21, "h": 0.30}, 
    "O_M3":   {"x": 0.61, "y": 0.62, "w": 0.18, "h": 0.31}
}

# 1. Bảng giá vẫn giữ nguyên để tra cứu tiền
# 1. DANH SÁCH NHÃN KHỚP 100% VỚI NÃO BỘ CỦA AI (TỪ INDEX 0 ĐẾN 10)
CLASS_FOODS = [
    'Canh chua',        # Index 0
    'Canh chua cá',     # Index 1
    'Canh rau',         # Index 2
    'Cá hú kho',        # Index 3
    'Cơm trắng',        # Index 4
    'Rau củ xào',       # Index 5
    'Sườn nướng',       # Index 6
    'Thịt kho',         # Index 7
    'Thịt kho trứng',   # Index 8
    'Trứng chiên',      # Index 9
    'Đậu hủ sốt cà'     # Index 10
]

# 2. BẢNG GIÁ ĐỒ ĂN (Tên key phải giống y hệt chữ có dấu ở mảng trên)
MENU_PRICES = {
    'Canh chua': 10000,
    'Canh chua cá': 25000,
    'Canh rau': 7000,
    'Cá hú kho': 30000,
    'Cơm trắng': 10000,
    'Rau củ xào': 10000,
    'Sườn nướng': 30000,
    'Thịt kho': 25000,
    'Thịt kho trứng': 30000,
    'Trứng chiên': 25000,
    'Đậu hủ sốt cà': 25000
}

@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('mo_hinh_.h5', compile=False)

model = load_ai_model()
# Chèn vào file Streamlit để kiểm tra nhãn trong não AI
st.write("Thứ tự món thực tế trong model:", model.output_shape)

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>HKT_Smart Canteen_</h1>", unsafe_allow_html=True)
st.markdown("---")

col_camera, col_bill = st.columns([1.4, 1])
detected_items = []
with col_camera:
    st.markdown("Camera / Tải ảnh")
    
    # Nguồn 1: Camera (Code cũ của bạn)
    camera_photo = st.camera_input("Đưa khay thức ăn vào khu vực này")
    
    # Nguồn 2: DÒNG CODE MỚI THÊM VÀO (Tải ảnh lên)
    uploaded_photo = st.file_uploader("Hoặc chọn ảnh khay cơm có sẵn", type=["jpg", "jpeg", "png"])
    
    # Gộp 2 nguồn: Ưu tiên ảnh chụp, nếu không có thì lấy ảnh tải lên
    img_data = camera_photo if camera_photo else uploaded_photo
    
    # Sửa lại điều kiện kiểm tra
    if img_data:
        with st.spinner("🤖 AI đang phân tích..."):
            # LƯU Ý: Sửa 'camera_photo.read()' thành 'img_data.read()' ở dòng dưới đây
            file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (640, 480))
            
            # Vẽ khung định vị lên ảnh hiển thị
            img_display = img_rgb.copy()
            for name, box in BOX_COORDS.items():
                x, y, w, h = int(box['x']*640), int(box['y']*480), int(box['w']*640), int(box['h']*480)
                cv2.rectangle(img_display, (x, y), (x + w, y + h), (50, 205, 50), 3)
                cv2.putText(img_display, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 205, 50), 2)
            
            st.image(img_display, caption="Ảnh chụp thực tế từ hệ thống", use_container_width=True)

            
            debug_logs = []
            
            for name, box in BOX_COORDS.items():
                x, y, w, h = int(box['x']*640), int(box['y']*480), int(box['w']*640), int(box['h']*480)
                cropped = img_rgb[y:y+h, x:x+w]
                if cropped.size == 0: continue
                
                resized = cv2.resize(cropped, (224, 224))
                
              
                input_tensor = np.expand_dims(preprocess_input(np.array(resized, dtype=np.float32)), axis=0) 
                
                
                preds = model.predict(input_tensor, verbose=0)
                max_idx = np.argmax(preds)
                score = np.max(preds)
                
                raw_name = CLASS_FOODS[max_idx]
                clean_name = raw_name.replace('_', ' ').title()                
                debug_logs.append(f"📍 **{name}**: Đoán là `{clean_name}` | Độ tự tin: `{score*100:.1f}%`")
                
                if score > 0.5:
                    detected_items.append({"Món": clean_name, "Giá": MENU_PRICES[raw_name]})
            
            with st.expander("🤖 NHẬT KÝ PHÂN TÍCH CỦA AI (XEM ĐỘ TỰ TIN %)", expanded=False):
                for log in debug_logs:
                    st.write(log)

with col_bill:
    st.markdown("### 🧾 Hóa Đơn Chi Tiết")
    
  
    st.markdown("""
    <style>
    .bill-box { background-color: #f9f9f9; color: #111111; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .item-row { display: flex; justify-content: space-between; font-size: 18px; margin-bottom: 10px; border-bottom: 1px dashed #ccc; color: #111111;}
    .total-row { display: flex; justify-content: space-between; font-size: 24px; font-weight: bold; color: #d32f2f; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

    tong_tien = sum(item["Giá"] for item in detected_items) if detected_items else 0
    
    html_bill = '<div class="bill-box">'
    if detected_items:
        for item in detected_items:
            html_bill += f'<div class="item-row"><span>🍽️ {item["Món"]}</span><span>{item["Giá"]:,} đ</span></div>'
    else:
        # Hóa đơn trống (dành cho lúc chưa chụp)
        html_bill += f'<div class="item-row" style="color: #888;"><span>Đang chờ khay cơm...</span><span>0 đ</span></div>'
        
    html_bill += f'<div class="total-row"><span>TỔNG CỘNG:</span><span>{tong_tien:,} VNĐ</span></div>'
    html_bill += '</div>'
    st.markdown(html_bill, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Đoạn chọn phương thức thanh toán và tính giảm giá ---
    hinh_thuc_tt = st.radio(
        "Chọn phương thức thanh toán",
        options=["Chuyển khoản ngân hàng", "Ví MoMo (Giảm 5%)"],
        horizontal=True
    )
    
    giam_gia = int(tong_tien * 0.05) if "Ví MoMo" in hinh_thuc_tt else 0
    so_tien_cuoi = tong_tien - giam_gia
    
    if giam_gia > 0:
        st.markdown(f"🔥 *Ưu đãi MoMo (5%):* `- {giam_gia:,} đ`")
    # -----------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Hãy Quét QR Tại Đây")
    
    # Tự động đổi ảnh QR tương ứng giữa MoMo và Ngân hàng
    if "Ví MoMo" in hinh_thuc_tt:
        file_anh_qr = "MOMO_QR.jpg"
    else:
        file_anh_qr = "MAQR.jpg"
        
    try: 
        # Đặt width=280 để ảnh QR hiện vừa vặn, cân đối trong khung hóa đơn
        st.image(file_anh_qr, width=280) 
    except: 
        st.error(f"⚠️ Thiếu ảnh QR '{file_anh_qr}' trong thư mục code!")
        
    # Hiển thị số tiền cuối sau khi đã tính toán giảm giá
    st.markdown(f"<p style='font-size: 20px; font-weight: bold; color: #111;'>Số tiền: {so_tien_cuoi:,} đ</p>", unsafe_allow_html=True)
    st.button("💵 Xác Nhận Giao Dịch", type="primary", use_container_width=True)