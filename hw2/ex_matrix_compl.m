% example: matrix completion with nuclear-norm regularization
%   minimize_X  F(X) = f(X) + Psi(X)
% where
%   f(X) = (1/|O|)\sum_{(i,j)\in O} (X_{ij} - S_{ij})^2 
%   Psi(X) = lambda*||X||_*  (weighted nuclear norm)

clear all; close all;

% load the rating matrix S
load('data/movie_rating');
figure; spy(S);

% subtract the scores by their mean
[i,j,s] = find(S);
linIdx = sub2ind(size(S),i,j);
S(linIdx) = S(linIdx) - sum(S(linIdx))/length(s); 

% construct mse function for the score matrix
f = func_matrix_mse(S);
% construct nuclear norm regularization function
lambda = 0.001;
Psi = func_nuclearnorm(lambda);

% set initial point
x0 = zeros(size(S));

% choose algorithmic options
opts.epsilon = 1e-8;
opts.maxitrs = 100;
opts.t_fixed = 10000;
%opts.linesearch = 'fixed';
opts.linesearch = 'bt';
opts.bt_init = 't_fixed';
%opts.bt_init = 'previous';
%opts.bt_init = 'adaptive';

% test different algorithms, results will be shown in figures
algms = {'proxgrad','Nesterov1stA','Nesterov1stB'};
[xs, Fs, ts] = test_algms(f, Psi, x0, opts, algms, 'matrix completion');


% display solution information
disp(' ');
disp('Rank of solution matrix and their MSE on observed scores');
disp('           Algorithm     rank    MSE')
disp('--------------------------------------------------------');
for i=1:length(algms)
    disp(sprintf('%20s %6i     %5.3f', algms{i}, rank(xs{i}), f.oracle(xs{i})));
end
disp(' ');