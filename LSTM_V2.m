clc;
clear all;
rng(42);  % 🔒 reproducibility

%% 🧾 1. Load data
data = readtable('Machine3_Data_Train.xlsx');

%% 🔮 2. Shift future values (60-minute forecast)

temp_future     = [data.temperature(61:end); NaN(60,1)];
pressure_future = [data.pressure(61:end); NaN(60,1)];
volt_future     = [data.volt(61:end); NaN(60,1)];

%% 🌡️ 3. Create Risk Levels
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

%% Risk Score
risk_score = temp_lvl + pressure_lvl + volt_lvl;

%% Risk Level
define_risk_level = @(x) ...
    (x <= 5) .* 1 + ...
    (x > 5 & x < 8) .* 2 + ...
    (x == 8) .* 3 + ...
    (x == 9) .* 4 + ...
    (x >= 10) .* 5;

risk_level = arrayfun(define_risk_level, risk_score);
%% ✅ 4. Filter rows
valid_idx = ~isnan(risk_level);
data = data(valid_idx, :);
risk_level = risk_level(valid_idx);
temp_lvl = temp_lvl(valid_idx);
pressure_lvl = pressure_lvl(valid_idx);
volt_lvl = volt_lvl(valid_idx);
risk_score = risk_score(valid_idx);

%% ⚙️ 5. Sliding Window Features
windowSize = 60;
numRows = height(data);
num_features = 3;
risk_valid_idx = find(~isnan(risk_level));
valid_idx = risk_valid_idx(risk_valid_idx > windowSize & risk_valid_idx + windowSize <= numRows);

num_samples = length(valid_idx);
X = zeros(num_samples, windowSize * num_features);
Y = zeros(num_samples, 1);

for j = 1:num_samples
    i = valid_idx(j) - windowSize;
    input_window = [ ...
        data.temperature(i+1:i+windowSize), ...
        data.pressure(i+1:i+windowSize), ...
        data.volt(i+1:i+windowSize) ...
    ];
    X(j, :) = reshape(input_window, 1, []);
    Y(j) = risk_level(valid_idx(j));
end

fprintf("✅ Sliding Window Created: %d samples, %d features\n", size(X,1), size(X,2));

%% 🧼 6. Split + Normalize
split_ratio = 0.7;
split_idx = round(split_ratio * size(X, 1));

XTrain = X(1:split_idx, :);
YTrain = Y(1:split_idx);
XTest = X(split_idx+1:end, :);
YTest = Y(split_idx+1:end);

%% 🧼 6. Normalize
mu = mean(XTrain);
sigma = std(XTrain);

XTrain = (XTrain - mu) ./ sigma;
XTest = (XTest - mu) ./ sigma;

%% 🤖 7. LSTM Parameters
classLabels = 1:5;
XTrainLSTM = num2cell(XTrain', [1]);
XTestLSTM = num2cell(XTest', [1]);
YTrainCat = categorical(YTrain, classLabels);
YTestCat  = categorical(YTest, classLabels);

layers = [
    sequenceInputLayer(size(XTrain,2))
    lstmLayer(128, 'OutputMode','last')   % ลดหน่วย LSTM
    dropoutLayer(0.3)
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(length(classLabels))
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 1, ...
    'ExecutionEnvironment','gpu');

%% 🔁 8. K-Fold Cross Validation (เลือกโมเดลที่ดีที่สุด)
k = 10;
cv = cvpartition(size(XTrain,1), 'KFold', k);
resultsTable = table();
bestAccuracy = -inf;
bestFold = -1;
best_net = [];

for fold = 1:k
    trainIdx = training(cv, fold);
    valIdx = test(cv, fold);
    
    XFold = num2cell(XTrain(trainIdx,:)', [1]);
    YFold = categorical(YTrain(trainIdx), classLabels);
    XV = num2cell(XTrain(valIdx,:)', [1]);
    YV = categorical(YTrain(valIdx), classLabels);
    
    net = trainNetwork(XFold, YFold, layers, options);
    
    YPred = classify(net, XV);
    YTrue = double(YV);
    YPred = double(YPred);
    
    cm = confusionmat(YTrue, YPred, 'Order', classLabels);
    TP = diag(cm);
    FP = sum(cm,1)' - TP;

    accuracy = sum(TP) / sum(cm(:));
    precision = mean(TP ./ (TP + FP + eps));
    
    resultsTable = [resultsTable; {fold, accuracy, precision}];

    if accuracy > bestAccuracy
        bestAccuracy = accuracy;
        bestFold = fold;
        best_net = net;
    end
end

resultsTable.Properties.VariableNames = {'Fold', 'Accuracy', 'Precision'};
disp(resultsTable);

fprintf('🎯 Best LSTM Model selected from Fold #%d | Accuracy: %.2f%%\n', bestFold, bestAccuracy * 100);

%% ✅ 9. Final Test using Best Model from K-Fold
final_net = best_net;  % ใช้โมเดลที่ดีที่สุดจาก Cross-Validation

YPredTest = classify(final_net, XTestLSTM);
YPredTest = double(YPredTest);
classNames = {'Very Low', 'Low', 'Medium', 'High', 'Very High'};

% 💾 Save LSTM Model and Related Data
save('LSTM_Model.mat', ...
    'final_net', ...                        % Best LSTM network
    'mu', 'sigma', ...                      % Normalization factors
    'temp_lvl', 'pressure_lvl', 'volt_lvl', ...  % Levels
    'risk_level', 'risk_score', ...         % Labels
    'data');                                % Raw + preprocessed data

%% 📊 Confusion Matrix: Train -------------------------------------------------------------------------------------------------------
YPredTrain = classify(final_net, XTrainLSTM);
YPredTrain = double(YPredTrain);

cm_train = confusionmat(YTrain, YPredTrain, 'Order', classLabels);
TP_train = diag(cm_train);

figure;
h = heatmap(classNames, classNames, cm_train, ...
        'Title', '📊 Confusion Matrix: Train Set (Risk Level LSTM)', ...
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
        'Title', '📊 Confusion Matrix: Test Set (Risk Level LSTM)', ...
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

time_test = 1:length(YTest);  % ใช้ index แทนเวลา

figure;
scatter(time_test, y_actual_jittered, 30, 'b', 'filled', 'DisplayName', 'Actual');
hold on;
scatter(time_test, y_pred_jittered, 30, 'r', 'filled', 'DisplayName', 'Predicted');
xlabel('Time Index');
ylabel('Risk Level');
legend;
title('📊 Jittered Scatter Plot: Actual vs Predicted Risk Level (LSTM)');
grid on;

%% 📈 Line Plot: Risk Level Over Time
figure;
plot(time_test, YTest, '-ob', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(time_test, YPredTest, '-xr', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
xlabel('Time Index');
ylabel('Risk Level');
title('📈 Actual vs Predicted Risk Level Over Time (LSTM)');
legend('Location', 'best');
grid on;

%% 📊 Grouped Bar Chart: เปรียบเทียบจำนวนแต่ละคลาส
actual_counts = histcounts(YTest, [classLabels, Inf]);
predicted_counts = histcounts(YPredTest, [classLabels, Inf]);

figure;
bar(classLabels - 0.15, actual_counts, 0.3, 'b', 'DisplayName', 'Actual');
hold on;
bar(classLabels + 0.15, predicted_counts, 0.3, 'r', 'DisplayName', 'Predicted');
xticks(classLabels);
xticklabels(classNames);
xlabel('Class');
ylabel('Count');
title('📊 Actual vs Predicted Risk Level Distribution (Test Set - LSTM)');
legend;
grid on;

%% 💾 11. Export Predictions
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

nTest = length(YPredTest);
test_idx = height(T) - nTest + 1 : height(T);
if length(test_idx) == length(YPredTest)
    T.predicted_risk_level = NaN(height(T), 1);
    T.predicted_risk_level(test_idx) = YPredTest;
    
    isCorrectTest = double(YPredTest == YTest);
    T.correct_prediction = NaN(height(T), 1);
    T.correct_prediction(test_idx) = isCorrectTest;
else
    error('❌ test_idx size mismatch');
end

T_testOnly = T(test_idx, :);
writetable(T_testOnly, 'RiskData_LSTM_Predictions.xlsx');
