% example: minimizing log-sum-exp function using different algorithms
% f(x) = log( \sum_{i=1}^m exp(a_i'*x + b_i) )
clear all; close all;

% reset random generator
rng('default');
rng(546);

% generate random data A and b
m = 500;
n = 200;
A = randn(m,n);
b = randn(m,1);

% construct the objective function (a_i transpose is ith row of A)
f = func_logsumexp(A, b);

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
%algms = {'Newton', 'BFGS'};
algms = {'gradient', 'Newton', 'BFGS', 'optimal', 'optimal simple'};

session_name = 'logsumexp t\_fixed=1';
[x_all,f_all,t_all,n_all] = test_algms(f, x0, opts, algms, session_name);