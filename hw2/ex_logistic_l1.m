% example: minimizing a logistic loss function with l1 regularization
% f(x) = (1/m) \sum_{i=1}^m log(1 + exp(-a_i'*x)) + (lambda)*||x||_1
clear all; close all;

% load the A matrix, where each row is a_i in the logistic function
load('data/data_logistic');
[m, n] = size(A);

% construct objective function: l2-regularized logistic regression
lambda1 = 0.01;     % weight for l1 regularization
lambda2 = 0.1;      % weight for l2 regularization
%f = func_logistic(A);
f = func_logistic_l2(A, lambda2);
Psi = func_l1(lambda1);

% set initial point
x0 = zeros(n,1);

% choose algorithmic options
opts.epsilon = 1e-8;
opts.maxitrs = 1000;
opts.t_fixed = 0.1;
%opts.linesearch = 'fixed';
opts.linesearch = 'bt';
opts.bt_init = 't_fixed';
%opts.bt_init = 'previous';
%opts.bt_init = 'adaptive';

opts.subg_stepsize = 's_sqrt';

% test different algorithms, results will be shown in figures
%algms = {'Nesterov1stB'};
%algms = {'proxgrad', 'Nesterov1stA'};
%algms = {'proxgrad', 'Nesterov1stB','subgradient'};
algms = {'subgradient', 'proxgrad', 'Nesterov1stA', 'Nesterov1stB'};
[xs, Fs, ts] = test_algms(f, Psi, x0, opts, algms, 'logistic\_l1');

% display sparsity of solutions
disp(sprintf('\nNumber of nonzero entries in solution (of dimension %i)',n));
disp('------------------------------------')
for i=1:length(algms)
    disp(sprintf('%20s %8i nnzs', algms{i}, sum(xs{i}~=0)));
end
