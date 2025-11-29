clc;
clear all;
rng(42);  % 🔒 reproducibility

%% 🧾 1. Load data with selected features
data = readtable('Machine3_Data_Train.xlsx');

%% 🔮 2. Shift future values (60-minute forecast)

temp_future     = [data.temperature(61:end); NaN(60,1)];
pressure_future = [data.pressure(61:end); NaN(60,1)];
volt_future     = [data.volt(61:end); NaN(60,1)];

%% 🌡️ 3. Create Risk Levels from future values
temp_lvl = (temp_future < 25) * 1 + ...
           (temp_future >= 25 & temp_future < 30) * 2 + ...
           (temp_future >= 30 & temp_future < 38) * 3 + ...
           (temp_future >= 38) * 4;

pressure_lvl = (pressure_future < 6) * 1 + ...
               (pressure_future >= 6 & pressure_future < 7.5) * 2 + ...
               (pressure_future >= 7.5 & pressure_future < 9) * 3 + ...
               (pressure_future >= 9) * 4;

volt_lvl = (volt_future == 0) * 1 + ...
           (volt_future > 0 & volt_future < 380) * 1 + ...
           (volt_future > 380 & volt_future < 410) * 3 + ...
           (volt_future >= 410) * 4;

risk_score = temp_lvl + pressure_lvl + volt_lvl;

define_risk_level = @(x) ...
    (x <= 5) .* 1 + ...
    (x > 5 & x < 8) .* 2 + ...
    (x == 8) .* 3 + ...
    (x == 9) .* 4 + ...
    (x >= 10) .* 5;

risk_level = arrayfun(define_risk_level, risk_score);

%% ✅ 4. Filter valid rows (complete targets)
valid_idx = ~isnan(risk_level);
data = data(valid_idx, :);
risk_level = risk_level(valid_idx);
temp_lvl = temp_lvl(valid_idx);
pressure_lvl = pressure_lvl(valid_idx);
volt_lvl = volt_lvl(valid_idx);
risk_score = risk_score(valid_idx);

%% ⚡️ 5. Sliding Window Feature Creation (Optimized Version)
windowSize = 60;
numRows = height(data);
num_features = 3;  % temp, pressure, volt

% 🔍 สร้าง index ที่ค่าของ risk_level มีค่า และอยู่ในช่วงที่สามารถสร้าง window ได้
risk_valid_idx = find(~isnan(risk_level));
valid_idx = risk_valid_idx(risk_valid_idx > windowSize & risk_valid_idx + windowSize <= numRows);

% ⚙️ เตรียม X/Y matrix ด้วยขนาดที่พอดี
num_samples = length(valid_idx);
X = zeros(num_samples, windowSize * num_features);
Y = zeros(num_samples, 1);

% 🔁 วนแบบรวดเร็ว (ไม่มี continue, ไม่มี reshape ซ้ำ)
for j = 1:num_samples
    i = valid_idx(j) - windowSize;
    input_window = [ ...
        data.temperature(i+1:i+windowSize), ...
        data.pressure(i+1:i+windowSize), ...
        data.volt(i+1:i+windowSize) ...
    ];
    X(j, :) = reshape(input_window, 1, []);
    Y(j, :) = risk_level(valid_idx(j));
end

fprintf("✅ Fast Sliding Window Done! Samples: %d | Features per sample: %d\n", size(X,1), size(X,2));

%% 🔀 4. แบ่ง Train/Test และ Normalize
% 🔀 Split + Normalize
split_ratio = 0.7;
split_idx = round(split_ratio * size(X, 1));

XTrain = X(1:split_idx, :);
YTrain = Y(1:split_idx);

XTest = X(split_idx+1:end, :);
YTest = Y(split_idx+1:end);

mu = mean(XTrain);
sigma = std(XTrain);
XTrain = (XTrain - mu) ./ sigma;
XTest = (XTest - mu) ./ sigma;

%% 🔧 5. ตั้งค่า SVM
svmParams.KernelFunction = 'rbf';
svmParams.KernelScale = 5;
svmParams.BoxConstraint = 50;

svmTemplate = templateSVM( ...
    'KernelFunction', svmParams.KernelFunction, ...
    'KernelScale', svmParams.KernelScale, ...
    'BoxConstraint', svmParams.BoxConstraint, ...
    'Standardize', false);

%% 🤖 6. K-Fold Cross Validation (Accuracy/Precision/ConfMat เท่านั้น)
k = 10;
cv = cvpartition(size(XTrain, 1), 'KFold', k);
resultsTable = table();
classLabels = [1 2 3 4 5];

bestAccuracy = -inf;
bestSVMModel = [];
bestFold = -1;  % เพิ่มตัวแปรเก็บ fold ที่ดีที่สุด

for fold = 1:k
    trainIdx = training(cv, fold);
    valIdx = test(cv, fold);
    XTrainFold = XTrain(trainIdx, :);
    YTrainFold = YTrain(trainIdx);
    XValidFold = XTrain(valIdx, :);
    YValidFold = YTrain(valIdx);

    svmModel = fitcecoc(XTrainFold, YTrainFold, 'Learners', svmTemplate);
    YPred = predict(svmModel, XValidFold);

    cm = confusionmat(YValidFold, YPred, 'Order', classLabels);
    TP = diag(cm);
    FP = sum(cm,1)' - TP;

    accuracy = sum(TP) / sum(cm(:));
    precision = mean(TP ./ (TP + FP + eps));

    resultsTable = [resultsTable; {fold, accuracy, precision}];

    if accuracy > bestAccuracy
        bestAccuracy = accuracy;
        bestSVMModel = svmModel;
        bestFold = fold;  % บันทึก fold ที่ดีที่สุด
    end
end
 
resultsTable.Properties.VariableNames = {'Fold', 'Accuracy', 'Precision'};
disp(resultsTable);


%% 🧠 7. ใช้โมเดลที่ดีที่สุดจาก Cross-Validation เป็นโมเดลสุดท้าย
finalSVM = bestSVMModel;

YPredTrainFinal = predict(finalSVM, XTrain);
acc_final_train = mean(YPredTrainFinal == YTrain);
fprintf('🎯 Final SVM Accuracy (Train Set จริง): %.2f%% [จาก Fold #%d]\n', acc_final_train * 100, bestFold);

% 💾 Save
save('SVM_Model.mat', ...
    'finalSVM', ...
    'svmTemplate', ...
    'mu', 'sigma', ...
    'temp_lvl', 'pressure_lvl', 'volt_lvl', ...
    'risk_level', 'risk_score', ...
    'data');

%% 📊 8. ทำนายและวัดผลบน Test Set
YPredTest = predict(finalSVM, XTest);
YPredTrain = predict(finalSVM, XTrain);
classNames = {'Very Low', 'Low', 'Medium', 'High', 'Very High'};

% 🔎 ตรวจว่าทำนายถูกหรือไม่ (ถูก = 1, ผิด = 0)
isCorrectTest = double(YPredTest == YTest);  % จะได้ vector 0/1

%% 📊 Confusion Matrix: Train -------------------------------------------------------------------------------------------------------
cm_train = confusionmat(YTrain, YPredTrain, 'Order', classLabels);
TP_train = diag(cm_train);

figure;
h = heatmap(classNames, classNames, cm_train, ...
        'Title', '📊 Confusion Matrix: Train Set (Risk Level SVM)', ...
        'XLabel', 'Predicted', ...
        'YLabel', 'Actual', ...
        'FontSize', 14, ...
        'Colormap', hot, ...
        'CellLabelColor', 'auto');
h.CellLabelFormat = '%.0f';

accuracy_train = sum(TP_train) / sum(cm_train(:));
precision_train = TP_train ./ (sum(cm_train,1)' + eps);

fprintf('\n✅ Overall Accuracy (Train): %.2f%%\n', accuracy_train * 100);
for i = 1:length(classNames)
    fprintf('%s → Precision: %.2f\n', ...
        classNames{i}, precision_train(i));
end

%% 📊 Confusion Matrix: Test
cm = confusionmat(YTest, YPredTest, 'Order', classLabels);
TP = diag(cm);

figure;
h = heatmap(classNames, classNames, cm, ...
        'Title', '📊 Confusion Matrix: Test Set (Risk Level SVM)', ...
        'XLabel', 'Predicted', 'YLabel', 'Actual', ...
        'FontSize', 14, 'Colormap', hot, ...
        'CellLabelColor', 'auto');
h.CellLabelFormat = '%.0f';

accuracy = sum(TP) / sum(cm(:));
precision = TP ./ (sum(cm,1)' + eps);

fprintf('\n✅ Overall Accuracy (Test): %.2f%%\n', accuracy * 100);
for i = 1:length(classNames)
    fprintf('%s → Precision: %.2f\n', ...
        classNames{i}, precision(i));
end

%% 📌 Jittered Scatter Plot: ค่าจริง vs ค่าทำนาย
jitterAmount = 0.05;
y_actual_jittered = YTest + (rand(size(YTest)) - 0.5) * jitterAmount;
y_pred_jittered = YPredTest + (rand(size(YPredTest)) - 0.5) * jitterAmount;

% ⚠️ หากไม่มี time_test ให้ใช้ index แทน
time_test = 1:length(YTest);

figure;
scatter(time_test, y_actual_jittered, 30, 'b', 'filled', 'DisplayName', 'Actual');
hold on;
scatter(time_test, y_pred_jittered, 30, 'r', 'filled', 'DisplayName', 'Predicted');
hold off;
xlabel('Time Index');
ylabel('Risk Level');
legend;
title('📊 Jittered Scatter Plot: Actual vs Predicted Risk Level');
grid on;

%% 📈 Line Plot: Risk Level Over Time
figure;
plot(time_test, YTest, '-ob', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(time_test, YPredTest, '-xr', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
xlabel('Time Index');
ylabel('Risk Level');
title('📈 Actual vs Predicted Risk Level Over Time');
legend('Location', 'best');
grid on;

%% 📊 Grouped Bar Chart: เปรียบเทียบจำนวนแต่ละคลาส
actual_counts = histcounts(YTest, [classLabels, Inf]);
predicted_counts = histcounts(YPredTest, [classLabels, Inf]);

figure;
bar(classLabels - 0.15, actual_counts, 0.3, 'b', 'DisplayName', 'Actual');
hold on;
bar(classLabels + 0.15, predicted_counts, 0.3, 'r', 'DisplayName', 'Predicted');
hold off;
xticks(classLabels);
xticklabels(classNames);
xlabel('Class');
ylabel('Count');
title('📊 Actual vs Predicted Risk Level Distribution (Test Set)');
legend;
grid on;

%% 🧾 Export ข้อมูล Risk Level พร้อม Score และค่าพยากร
T = table();
if ismember('sendDate', data.Properties.VariableNames)
    T.sendDate = data.sendDate;
end
T.temperature    = data.temperature;
T.pressure       = data.pressure;
T.volt           = data.volt;
T.temp_lvl       = temp_lvl;
T.pressure_lvl   = pressure_lvl;
T.volt_lvl       = volt_lvl;
T.risk_score     = risk_score;
T.risk_level     = risk_level;

% 🔮 เพิ่มค่าพยากรของโมเดลลงในตาราง
nTest = length(YPredTest);  % ขนาดของ Test Set
test_idx = height(T) - nTest + 1 : height(T);  % คำนวณตำแหน่งท้ายสุด

% ตรวจสอบความถูกต้องก่อนใส่
if length(test_idx) == length(YPredTest)
    T.predicted_risk_level = NaN(height(T), 1);
    T.predicted_risk_level(test_idx) = YPredTest;

    % ✅ เพิ่มคอลัมน์ที่บอกว่าทำนายถูกหรือไม่ (0/1)
    isCorrectTest = double(YPredTest == YTest);  % แปลง logic → 0/1
    T.correct_prediction = NaN(height(T), 1);    % เตรียม column
    T.correct_prediction(test_idx) = isCorrectTest;
else
    error('❌ ขนาดไม่ตรง: test_idx = %d, YPredTest = %d', length(test_idx), length(YPredTest));
end

% ✂️ Export เฉพาะ Test Set
T_testOnly = T(test_idx, :);
writetable(T_testOnly, 'RiskData_WithPredictions.xlsx');


