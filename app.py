import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. CẤU HÌNH & TẢI DỮ LIỆU --- #
st.set_page_config(page_title="Dự đoán Tiểu đường", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv('diabetes_data.csv')
    return data

df = load_data()
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Chuẩn hóa dữ liệu #
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Huấn luyện mô hình (Dùng cache_resource để app không phải load lại mô hình mỗi khi nhập số) #
@st.cache_resource
def train_model(_X_scaled, _y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(_X_scaled, _y)
    return model

model = train_model(X_scaled, y)

# --- 2. GIAO DIỆN CHÍNH --- #
st.title("🩺 Hệ thống Tham chiếu & Dự đoán Tiểu đường")
st.markdown("---")

# Sidebar: Nhập liệu với Tooltips giải thích #
st.sidebar.header("📥 Nhập chỉ số của bạn")
def user_input():
    preg = st.sidebar.number_input('1. Số lần mang thai', 0, 20, 1)
    glu = st.sidebar.number_input('2. Glucose (sau 2h)', 0, 300, 120, help="Bình thường dưới 140 mg/dL")
    bp = st.sidebar.number_input('3. Huyết áp tâm trương (mm Hg)', 0, 150, 70)
    skin = st.sidebar.number_input('4. Độ dày nếp gấp da (mm)', 0, 100, 20)
    ins = st.sidebar.number_input('5. Insulin (mu U/ml)', 0, 900, 80)
    bmi = st.sidebar.number_input('6. Chỉ số BMI', 0.0, 70.0, 25.0, help="BMI = Cân nặng / (Chiều cao^2)")
    pedi = st.sidebar.number_input('7. Chức năng phả hệ (0.0 - 2.5)', 0.0, 2.5, 0.5, help="Chỉ số nguy cơ di truyền")
    age = st.sidebar.number_input('8. Độ tuổi', 1, 120, 30)
    
    features = pd.DataFrame([[preg, glu, bp, skin, ins, bmi, pedi, age]], columns=X.columns)
    return features, bmi

input_df, user_bmi = user_input()

# --- 3. PHÂN TÍCH & HIỂN THỊ --- #
if st.button('🚀 Phân tích kết quả'):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    st.divider()
    
    # Hiển thị Metrics quan trọng #
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Chỉ số BMI", user_bmi, delta="Cần chú ý" if user_bmi >= 25 else "Bình thường", delta_color="inverse")
    with col_m2:
        res_text = "Dương tính" if prediction[0] == 1 else "Âm tính"
        st.metric("Kết quả dự đoán", res_text)
    with col_m3:
        score = probability[0][prediction[0]] * 100
        st.metric("Độ tin cậy", f"{score:.2f}%")

    # Chi tiết kết quả #
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Đánh giá chi tiết")
        if prediction[0] == 1:
            st.error("⚠️ Cảnh báo: Các chỉ số của bạn có sự tương đồng lớn với nhóm bệnh nhân tiểu đường trong dữ liệu mẫu.")
        else:
            st.success("✅ Kết quả khả quan: Các chỉ số của bạn hiện đang nằm trong ngưỡng an toàn so với dữ liệu tham chiếu.")
            
    with col2:
        st.subheader("📈 Biểu đồ so sánh (Của bạn vs. Trung bình)")
        comparison_df = pd.concat([input_df, pd.DataFrame([X.mean()], columns=X.columns)], ignore_index=True)
        comparison_df.index = ['Của bạn', 'Trung bình mẫu']
        # Vẽ biểu đồ cột #
        st.bar_chart(comparison_df.T)

    # Bảng dữ liệu thô #
    with st.expander("Xem bảng số liệu chi tiết"):
        st.table(comparison_df)

st.info("💡 Lưu ý: Đây là hệ thống tham chiếu dựa trên thuật toán, kết quả chỉ mang tính chất tham khảo, không thay thế chẩn đoán của bác sĩ.")
##muhehehehe
