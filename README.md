# DiabetesPrediction
Diabetes Prediction System
Dự đoán nguy cơ tiểu đường dựa trên thuật toán Học máy (KNN & Logistic Regression)

  Giới thiệu Đồ án
Dự án tập trung vào việc xây dựng một hệ thống sàng lọc sớm bệnh tiểu đường dựa trên bộ dữ liệu Pima Indians Diabetes. 
Hệ thống cho phép người dùng nhập các chỉ số sinh học cơ bản và nhận kết quả cảnh báo ngay lập tức.

  Tính năng chính
Phân tích dữ liệu (EDA): Xử lý dữ liệu thô, xử lý giá trị thiếu (Missing values) bằng phương pháp Median.

Mô hình dự đoán: Sử dụng thuật toán K-Nearest Neighbors (KNN) được chuẩn hóa qua StandardScaler.

Giao diện trực quan: Ứng dụng Web được xây dựng bằng Streamlit, cho phép nhập liệu và hiển thị kết quả thời gian thực.

Báo cáo so sánh: So sánh chỉ số cá nhân với giá trị trung bình của cộng đồng.

  Công nghệ & Thư viện sử dụng
Ngôn ngữ chính: Python 3.x

Giao diện người dùng: Streamlit

Thư viện Học máy: Scikit-learn (Sử dụng KNeighborsClassifier)

Xử lý & Phân tích dữ liệu: Pandas, Numpy

Tiền xử lý: StandardScaler (Chuẩn hóa dữ liệu đầu vào)

Quy trình xử lý kỹ thuật
Tải dữ liệu: Đọc file diabetes_data.csv và xử lý ngoại lệ nếu thiếu file.

Chuẩn hóa (Scaling): Sử dụng StandardScaler để đưa 8 chỉ số sức khỏe về cùng một thang đo, giúp thuật toán KNN tính toán khoảng cách chính xác hơn.

Huấn luyện mô hình: Sử dụng thuật toán K-Nearest Neighbors (KNN) với n_neighbors=5.

Dự đoán & Phân tích: * Tính toán xác suất (Probability) để đưa ra độ tin cậy tham chiếu.

So sánh chỉ số người dùng với giá trị trung bình (Mean) của tập mẫu để đưa ra cái nhìn tổng quan.

Hướng dẫn cài đặt và chạy Local
Để chạy dự án này trên máy tính cá nhân, bạn thực hiện các bước sau:

Clone repository:
  git clone https://github.com/PHAsnezto/DiabetesPrediction.git
  cd DiabetesPrediction

Cài đặt thư viện cần thiết:
  pip install streamlit pandas scikit-learn

Khởi chạy ứng dụng:
  streamlit run app.py
  
  Các chỉ số đầu vào (Features)
Hệ thống sử dụng 8 thông số đầu vào quan trọng:

Số lần mang thai (Pregnancies)

Chỉ số Glucose (sau 2h)

Huyết áp tâm trương (Blood Pressure)

Độ dày nếp gấp da (Skin Thickness)

Chỉ số Insulin

Chỉ số khối cơ thể (BMI)

Chức năng phả hệ tiểu đường (Diabetes Pedigree Function)

Độ tuổi (Age)

Thành viên thực hiện
  Hoàng Bảo Minh (Trưởng nhóm), Phạm Hồng Anh, Vương Nhĩ Khang
