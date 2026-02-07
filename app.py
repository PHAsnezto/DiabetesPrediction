import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Cấu hình trang
st.set_page_config(page_title="Dự đoán Tiểu đường", layout="wide")

st.title("🩺 Hệ thống Tham chiếu & Dự đoán Tiểu đường")
st.write("Dựa trên dữ liệu thực tế từ 8 chỉ số sức khỏe để đưa ra cảnh báo.")

# 1. Hàm tải dữ liệu
@st.cache_data
def load_data():
    try:
        # Thay 'diabetes_data.csv' bằng tên file chính xác của bạn
        data = pd.read_csv('diabetes_data.csv')
        return data
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ Không tìm thấy file 'diabetes_data.csv'. Vui lòng kiểm tra lại trên GitHub!")
    st.stop()

# 2. Chuẩn bị dữ liệu tham chiếu
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Chuẩn hóa dữ liệu (giúp việc so sánh các chỉ số công bằng hơn)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Huấn luyện mô hình tham chiếu (KNN)
model = KNeighborsClassifier(n_neighbors=5) # Tìm 5 người có chỉ số gần nhất
model.fit(X_scaled, y)

# 3. Giao diện nhập liệu
st.sidebar.header("📥 Nhập chỉ số của bạn")
def user_input():
    # Sử dụng đúng 8 chỉ số bạn đã cung cấp
    preg = st.sidebar.number_input('1. Số lần mang thai', 0, 20, 1)
    glu = st.sidebar.number_input('2. Glucose (sau 2h)', 0, 300, 120)
    bp = st.sidebar.number_input('3. Huyết áp tâm trương (mm Hg)', 0, 150, 70)
    skin = st.sidebar.number_input('4. Độ dày nếp gấp da (mm)', 0, 100, 20)
    ins = st.sidebar.number_input('5. Insulin (mu U/ml)', 0, 900, 80)
    bmi = st.sidebar.number_input('6. Chỉ số BMI', 0.0, 70.0, 25.0)
    pedi = st.sidebar.number_input('7. Chức năng phả hệ (0.0 - 2.5)', 0.0, 2.5, 0.5)
    age = st.sidebar.number_input('8. Độ tuổi', 1, 120, 30)
    
    features = pd.DataFrame([[preg, glu, bp, skin, ins, bmi, pedi, age]], 
                            columns=X.columns)
    return features

input_df = user_input()

# 4. Thực hiện dự đoán
if st.button('Phân tích kết quả'):
    # Chuẩn hóa dữ liệu người dùng nhập
    input_scaled = scaler.transform(input_df)
    
    # Dự đoán
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Kết quả phân tích")
        if prediction[0] == 1:
            st.warning("⚠️ Cảnh báo: Bạn có các chỉ số tương đồng với nhóm người mắc bệnh.")
        else:
            st.success("✅ Chúc mừng: Các chỉ số của bạn hiện nằm trong nhóm an toàn.")
            
    with col2:
        st.subheader("📈 Độ tin cậy tham chiếu")
        score = probability[0][prediction[0]] * 100
        st.info(f"Mức độ tương đồng với dữ liệu mẫu: **{score:.2f}%**")

    # Hiển thị bảng so sánh
    st.subheader("🔍 So sánh chỉ số của bạn với trung bình cộng")
    comparison_df = pd.concat([input_df, pd.DataFrame([X.mean()], columns=X.columns)], ignore_index=True)
    comparison_df.index = ['Của bạn', 'Trung bình cộng mẫu']
    st.table(comparison_df)
