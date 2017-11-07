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
[normalized_x, mu, sigma] = featureNormalize(X);  % Normalize


% Plot data
plot(X, y(:, 4));
xlabel('Date in seconds from Jan 1, 1970 (Unix Time)')
ylabel('Closing price in USD')

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

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
