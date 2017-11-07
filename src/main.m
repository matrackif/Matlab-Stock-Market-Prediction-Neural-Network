%% Initialization
clear ; close all; clc

%% Setup the parameters 
input_layer_size  = 1;  % date
hidden_layer_size = 5;   % 5 hidden units
num_labels = 4;          % 5 outputs open, high, low, close and volume (if num_labels is 4 we ignore volume)
                          % (note that we have mapped "0" to label 10)
input_dates_file = 'dates_daily_CDR.csv';
stock_attributes_file = 'results_daily_CDR.csv';
% Did this to prevent Matlab from cutting off decimal places for every
% column except the volume column
format longE; 
X = csvread(input_dates_file);
y = csvread(stock_attributes_file);
% [normalized_x, mu, sigma] = featureNormalize(X);  % Normalize

% Plot data
plot(X, y(:, 4));
xlabel('Date in seconds from Jan 1, 1970 (Unix Time)')
ylabel('Closing price in USD')
%% =========== Learning Curve =============
%

% Partition X into cross validation set, training set, and test set
training_set_size = (0.6 * size(X, 1));
cross_validation_size = (0.2 * size(X, 1));
test_set_size = (0.2 * size(X, 1));
% TODO cast to integer?
cv = X((training_set_size + 1):(training_set_size + cross_validation_size), :);
cv_y = y((training_set_size + 1):(training_set_size + cross_validation_size), :);
test = X((training_set_size + cross_validation_size + 1):(training_set_size + cross_validation_size + cross_validation_size), :);
test_y = y((training_set_size + cross_validation_size + 1):(training_set_size + cross_validation_size + cross_validation_size), :);
X = X(1:training_set_size, :);
y = y(1:training_set_size, :);
[normalized_x, mu, sigma] = featureNormalize(X);

lambda = 0;
% We do not normalize when finding the learning curve? According to Andrew
% NG's example
[error_train, error_val] = ...
    learningCurve(X, y, ...
                  cv, cv_y, ...
                  input_layer_size, hidden_layer_size, num_labels, ...
                  lambda);
              
plot(1:training_set_size, error_train, 1:training_set_size, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
% axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
% for i = 1:m
    % fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
% end

%%%%%%%%%%
lambda = 3;
[nn_params, cost] = train(normalized_x, y, input_layer_size, hidden_layer_size, num_labels, lambda)

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, normalized_x);

last_day_in_training_set = X(1, :);
one_day_ahead = last_day_in_training_set + (3600 * 24);
three_days_ahead = last_day_in_training_set + (3600 * 24 * 3);
one_week_ahead = last_day_in_training_set + (3600 * 24 * 7);

future = [one_day_ahead; three_days_ahead; one_week_ahead];
% Normalize according to mu and sigma obtained from training set
future = bsxfun(@minus, future, mu);
future = bsxfun(@rdivide, future, sigma); 
future_pred = predict(Theta1, Theta2, future);

% Plot prediction
plot(normalized_x, y(:, 4), normalized_x, pred(:, 4));
title('Predction vs Actual Prices')
legend('Training Set', 'Prediction')
xlabel('Date (Normalized)')
ylabel('Closing Price')
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
