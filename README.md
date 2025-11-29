🔧 (1) LSTM-Based Predictive Maintenance Risk Classification
Forecast machine risk levels 60 minutes ahead using Deep Learning (LSTM)
📌 Overview
โปรเจกต์นี้ออกแบบมาเพื่อ พยากรณ์ระดับความเสี่ยงของเครื่องจักร (Risk Level 1–5) ล่วงหน้า 60 นาที โดยใช้ข้อมูลเซนเซอร์จาก Industrial Air Compressor เช่น

- Temperature
- Pressure
- Voltage

ทำงานแบบ Supervised Learning → Feature Engineering → LSTM Deep Learning → Evaluation → Export Report

🎯 Objective
ทำนายความเสี่ยงล่วงหน้า 1 ชั่วโมง
ใช้ Sliding Window 60 นาที → 180 features
ประเมินด้วย 10-fold cross-validation
เลือกโมเดลที่ดีที่สุดและทดสอบกับ Test Set
ส่งออกไฟล์ Excel ใช้งานจริงในโรงงาน

🧠 Core Method
1) Shift Target 60 นาที
temp_future = [data.temperature(61:end); NaN(60,1)];
pressure_future = [data.pressure(61:end); NaN(60,1)];
volt_future = [data.volt(61:end); NaN(60,1)];

2) สร้าง Risk Level
คำนวณ level ของ temp/pressure/volt
รวมคะแนนเป็น risk_score
Map → Risk Level 1–5

4) Sliding Window (60 × 3 features)
   
Feature size = 180 features / sample
input_window = [
    data.temperature(i+1:i+windowSize), ...
    data.pressure(i+1:i+windowSize), ...
    data.volt(i+1:i+windowSize)
];

5) Normalize + Split
Train 70% / Test 30%
Normalize ด้วยเฉพาะค่าจาก Train

🤖 LSTM Model Architecture
layers = [
    sequenceInputLayer(180)
    lstmLayer(128,"OutputMode","last")
    dropoutLayer(0.3)
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer
];

Optimizer: Adam
Epoch: 100
Batch Size: 256
Cross-Validation: 10-fold

📊 Evaluation
Confusion Matrix (Train/Test)
Accuracy, Precision per class
Risk Trend Plot
Distribution Plot
Scatter Plot of Predictions
บันทึกโมเดลที่ดีที่สุด:

final_net = best_net;
save('LSTM_Model.mat');

💾 Export Excel
writetable(T_testOnly, 'RiskData_LSTM_Predictions.xlsx');


ไฟล์ประกอบด้วย:
temperature / pressure / volt
temp_lvl / pressure_lvl / volt_lvl
risk_score
actual vs predicted risk_level
correct_prediction

⭐ Highlights
ทำ Deep Learning เต็มกระบวนการ
ใช้จริงกับโรงงาน (Industrial AI)
Cross-validation → วัดประสิทธิภาพชัด
Export ให้ใช้งานจริงในโรงงาน

🔧 (2) SVM-Based Predictive Maintenance Risk Classification
Predict machine risk levels using SVM + Feature Engineering

📌 Overview
โปรเจกต์นี้ใช้ Support Vector Machine (SVM) เพื่อจำแนกระดับความเสี่ยงของเครื่องจักร (1–5)
เหมาะกับข้อมูลที่ไม่เยอะมาก และมีโครงสร้างเชิงเส้น/ไม่เชิงเส้นร่วมกัน

🎯 Objective
จำแนก Risk Level 1–5
ใช้ Feature Engineering จาก temp/pressure/volt
Test ด้วยไฟล์จริงแยก Train/Test
10-fold cross-validation เพื่อเลือกโมเดลที่ดีที่สุด

🧠 Core Method
1) Feature Engineering
Mean temperature
Max pressure
Voltage variation
Moving average
Rolling difference
รวมเป็น feature vector สำหรับแต่ละช่วงเวลา

2) Risk Mapping
คล้าย LSTM แต่เป็น single timestamp
temp → level
pressure → level
volt → level
รวมคะแนน → map risk_level (1–5)

3) Normalize + Split
Train 70%
Test 30%
Standardization: (x - mean) / std
(ใช้ mean/std จาก Train เท่านั้น)

🤖 SVM Model
ใช้ SVM (RBF Kernel) เพราะข้อมูลมีความซับซ้อนและ boundary ไม่เป็นเส้นตรง
model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True
)
model.fit(X_train, y_train)

📊 Evaluation Metrics
Accuracy
Precision, Recall per class
Confusion Matrix
ROC Curve per class (One-vs-Rest)
Prediction Probability

💾 Export Excel
df.to_excel("SVM_Risk_Predictions.xlsx", index=False)

รวมผลลัพธ์:
Features
Actual Risk
Predicted Risk
Confidence Score

⭐ Highlights
ใช้ SVM พร้อม Feature Engineering
ประเมินด้วย 10-fold cross-validation
ใช้ข้อมูลจริงจากโรงงาน
ง่ายต่อการ Deploy ร่วมกับระบบ ERP
ทดสอบกับชุด Train/Test แยกไฟล์ชัดเจน
