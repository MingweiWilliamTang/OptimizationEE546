% example: minimizing a logistic loss function with l2 regularization
% f(x) = (1/m) \sum_{i=1}^m log(1 + exp(-a_i'*x)) + (lambda/2)*||x||_2^2
clear all; close all;

% load the A matrix, where each row is a_i in the logistic function
load data_logistic;
[m, n] = size(A);

% construct objective function: l2-regularized logistic regression
lambda = 0;
f = func_logistic_l2(A, lambda);

% set initial point
x0 = zeros(n,1);

% choose algorithmic options
opts.epsilon = 1e-8;
opts.maxitrs = 300;
opts.t_fixed = 1;
%opts.linesearch = 'fixed';
opts.linesearch = 'bt';
opts.bt_init = 't_fixed';
%opts.bt_init = 'previous';
%opts.bt_init = 'adaptive';

% test different algorithms, results will be shown in figures
%algms = {'gradient', 'Newton', 'BFGS', 'optimal', 'optimal simple'};
algms = {'gradient', 'BFGS', 'optimal', 'optimal simple'};
%algms = {'Newton'};
test_algms(f, x0, opts, algms, 'logistic\_l2');
