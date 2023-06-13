% prediction of COVID-19 daily infections in Israel
clear; close all; clc;


%% Load data

raw_data = readtable('corona_new_cases.csv');

%% Preprocess data

n_days = 14;

X_all   = buffer(raw_data{:, 'New_verified_cases'}, n_days, n_days - 1);
X_all   = X_all(:, n_days:(end - 1));
Y0_all  = raw_data{(n_days + 1):end, 'New_verified_cases'}';

n_samples = length(Y0_all);

%% Apply a logarithmic transformation to all data

X_all   = log(X_all);
Y0_all  = log(Y0_all);

%% Split data into train and validation sets

valid_ratio = 0.2;

n_train_samples = floor((1 - valid_ratio)*n_samples);
X_train         = X_all(:, 1:n_train_samples);
Y0_train        = Y0_all(1:n_train_samples);
X_valid         = X_all(:, (n_train_samples + 1):end);
Y0_valid        = Y0_all((n_train_samples + 1):end);

%% Define the network

% Set dimensions
n_input     = n_days;
n_hidden    = 70;
n_output    = 1;

% Initialize weights
W = randn(n_hidden, n_input);
J = randn(n_output, n_hidden);

% Set an activation function for each layer
% examin the activation functions code and understand their output
g1 = @ReLU;
g2 = @Linear;

%% Declare the learning parameters

eta      	= 0.0001 ; 
n_epochs    =30000;

%% Learn

% Loop over learning epochs
for ep = 1:n_epochs
    
    % random order of samples
    samp_order = randperm(n_train_samples);
    
    % Loop over all samples
    for samp = 1:n_train_samples
    
        % Choose a random sample
        s   = samp_order(samp);
        x   = X_train(:, s);
        y0  = Y0_train(s);
       
        % Forward pass
        [h, hp] = g1(W*x);  %  get the first layer value and derivative for 
                    %       the current example
        [y, yp] = g2(J*h);  %  get the output layer value and derivative for 
                    %      the current example
        
        % Backward pass
        delta2 =(y-y0)*yp;      % calculate the delta for the J weights 
        dJ =(-eta*delta2*h)';      % implement the online update rule for J weights
        delta1 = delta2*(J.*hp');   % calculate the delta for the W weights 
        dW =-eta.*delta1'*x';    % implement the online update rule for W weights
        
        % Update weights
        W = W + dW;
        J = J + dJ;
        
    end
        
end

%% Get output

%  Forward pass the whole training set
h_train = g1(W*X_train); 
Y_train = g2(J*h_train); 

%  Forward pass the whole validation set
h_valid = g1(W*X_valid);
Y_valid = g2(J*h_valid);

%% Print errors and R2

% Squared errors
train_err =mean(0.5*((Y_train-Y0_train).^2));% calculate the mean squared error for the training set
valid_err =mean(0.5*((Y_valid-Y0_valid).^2));% calculate the mean squared error for the validation set

fprintf('Training error:\t%g\nValidation error:\t%g\n', ...
    train_err, valid_err);

fprintf('\n');

% "Undo" the loragithmic transformation
exp_Y_train = exp(Y_train);
exp_Y0_train = exp(Y0_train);
exp_Y_valid = exp(Y_valid);
exp_Y0_valid = exp(Y0_valid);

% R2
train_R2 = corr(exp_Y_train', exp_Y0_train');
valid_R2 = corr(exp_Y_valid', exp_Y0_valid');
fprintf('Training R²:\t%g\nValidation R²:\t%g\n', ...
    train_R2, valid_R2);

%% Plot fit

% plot the predicted values of the train and validation vs. the
% true values. Use a scatter plot.  
% 1. use differant colors for each set
% 2. create a figure legend with the labels for each set
% 3. create a dashed black line along the optimal distribution of the data
hold on
scatter(exp_Y0_train, exp_Y_train,'b','d');
scatter(exp_Y0_valid, exp_Y_valid,'m','d');
x_plot = linspace(0,max([exp_Y_train exp_Y0_train exp_Y0_valid exp_Y_valid]));
y_plot = linspace(0,max([exp_Y_train exp_Y0_train exp_Y0_valid exp_Y_valid]));
plot(x_plot,y_plot,'--k');
xlabel('Teachers Output');
ylabel('Network Output');
title('Learning Process','FontSize',16);
legend('Train','Valid','Optimal');
hold off
