function [x_all,f_all,t_all,n_all] = test_algms(f, x0, opts, algms, session)
% test different algorithms for smooth convex optimization
% Inputs:
%   f:    a function object that implements [fval, grad, Hess] = f.oracle(x)
%   x0:   the starting point for the iterative algorithm
%   opts: a struct of options and parameters for the algorithm
%   algms: a cell arry of algorithm names, for example, 
%               algms = {'gradient', 'Newton', 'optimal'}
%   session: a string name for this testing session 
% Output: 
%   x_all:  a cell array, each cell is final solution of an algorithm
%   f_all:  a cell array, each cell is vector of function values
%   t_all:  a cell array, each cell is vector of step sizes 
%   n_all:  a cell array, each cell is vector of number of line searches
%
% This functin also generates plots to visualize the solution histories.

if nargin < 5; session = ' '; end

N = length(algms);

x_all = cell(N,1);
f_all = cell(N,1);
t_all = cell(N,1);
n_all = cell(N,1);

fmax = -Inf;
fmin = Inf;
flen = 0;

disp('           Algorithm     time')
disp('------------------------------------')
% solve the same problem with different algorithms
for i=1:N
    tic;
    switch lower(algms{i})
        case 'gradient'
            [x, fs, ts, nls] = algm_gradient(f, x0, opts);
        case 'newton'
            [x, fs, ts, nls] = algm_Newton(f, x0, opts);
        case 'bfgs'
            [x, fs, ts, nls] = algm_BFGS(f, x0, opts);
        case 'bfgs inv'
            [x, fs, ts, nls] = algm_BFGS_inv(f, x0, opts);
        case 'optimal'
            [x, fs, ts, nls] = algm_optimal(f, x0, opts);
        case 'optimal simple'
            [x, fs, ts, nls] = algm_optimal_simple(f, x0, opts);
        otherwise
            error(['Algorithm' algms{i} 'is not implemented.']);
    end
    T = toc;
    
    % display solution time used by each algorithm
    disp(sprintf('%20s %8.2f sec', algms{i}, T)); 
    
    x_all{i} = x;
    f_all{i} = fs;
    t_all{i} = ts;
    n_all{i} = nls;
 
    fmax = max(fmax, max(fs));
    fmin = min(fmin, min(fs));
    flen = max(flen, length(fs));
end

%fmin

% use known fmin if it is known
%if isfield(opts, 'fmin'); fmin = opts.fmin; end

% define line specs for plot results
line_specs = {'b--', 'k-', 'r-.', 'g-', 'm-', 'c-', 'y-'};

% plot the objective gaps at each iteration
figure(1)
for i=1:N
    semilogy(f_all{i} - fmin, line_specs{i}, 'LineWidth', 2);
    hold on;
end
axis([0, flen, 1.0e-12, fmax-fmin]);
set(gca, 'FontSize', 12);
xlabel('k');
ylabel('f(x)-f*');
legend(algms{:});
title(session);
%saveas(1,'fig31','png');
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
%saveas(2,'fig32','png');
% plot number of line searches at each iteration
figure(3)
for i=1:N
    plot(n_all{i}, line_specs{i}, 'LineWidth', 2);
    hold on;
end
set(gca, 'FontSize', 12);
xlabel('k');
ylabel('# backtrackings');
legend(algms{:});
title(session);

%saveas(3,'fig33','png');