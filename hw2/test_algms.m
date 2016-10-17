function [x_all,F_all,t_all] = test_algms(f, Psi, x0, opts, algms, session)
% test different algorithms for solving composite convex optimization 
%   minimize { F(x) = f(x) + Psi(x) }
% Inputs: 
%   f:    a function object that implements [fval, grad] = f.oracle(x)
%   Psi:  a function object that implements proximal mapping
%   x0:   the starting point for the iterative algorithm
%   opts: a struct of options and parameters for the algorithm
%   algms: a cell arry of algorithm names, for example, 
%               algms = {'subgradient', 'prox grad', 'Nesterov 1st'}
%   session: a string name for this testing session 
% Output: 
%   x_all:  a cell array, each cell is final solution of an algorithm
%   F_all:  a cell array, each cell is vector of function values
%   t_all:  a cell array, each cell is vector of step sizes 
%
% This functin also generates plots to visualize the solution histories.

if nargin < 6; session = ' '; end

N = length(algms);

x_all = cell(N,1);
F_all = cell(N,1);
t_all = cell(N,1);

Fmax = -Inf;
Fmin = Inf;
Flen = 0;

disp('           Algorithm     time')
disp('------------------------------------')
% solve the same problem with different algorithms
for i=1:N
    tic;
    switch lower(algms{i})
        case 'subgradient'
            [x, Fs, ts] = algm_subgradient(f, Psi, x0, opts);
        case 'proxgrad'
            [x, Fs, ts] = algm_proxgrad(f, Psi, x0, opts);
        case 'nesterov1sta'
            [x, Fs, ts] = algm_Nesterov1stA(f, Psi, x0, opts);
        case 'nesterov1stb'
            [x, Fs, ts] = algm_Nesterov1stB(f, Psi, x0, opts);
        otherwise
            error(['Algorithm' algms{i} 'is not implemented.']);
    end
    T = toc;
    
    % display solution time used by each algorithm
    disp(sprintf('%20s %8.2f sec', algms{i}, T)); 
    
    x_all{i} = x;
    F_all{i} = Fs;
    t_all{i} = ts;
 
    Fmax = max(Fmax, max(Fs));
    Fmin = min(Fmin, min(Fs));
    Flen = max(Flen, length(Fs));
end

% define line specs for plot results
line_specs = {'k-', 'b--', 'r-.', 'g-', 'm-', 'c-', 'y-'};

% plot the objective gaps at each iteration
figure(1)
for i=1:N
    semilogy(F_all{i} - Fmin, line_specs{i}, 'LineWidth', 2);
    hold on;
end
axis([0, Flen, 1.0e-10, Fmax-Fmin]);
set(gca, 'FontSize', 12);
xlabel('k');
ylabel('F(x)-F*');
legend(algms{:});
title(session);
saveas(1,'fig61','png');
% plot the chosen step sizes at each iteraiton
figure(2)
for i=1:N
    plot(t_all{i}, line_specs{i}, 'LineWidth', 2);
    hold on;
end
set(gca, 'FontSize', 12);
xlabel('k');
ylabel('stepsize t_k');
legend(algms{:});
title(session);
saveas(2,'fig62','png');