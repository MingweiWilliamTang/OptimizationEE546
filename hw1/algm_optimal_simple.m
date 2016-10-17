function [x, fs, ts, nls] = algm_optimal_simple(f, x0, opts)
% Simple variant of Nesterov's optimal method for smooth convex optimization 
% Inputs: 
%   f:    a function object that implements [fval, grad, Hess] = f.oracle(x)
%   x0:   the starting point for the iterative algorithm
%   opts: a struct of options and parameters for the algorithm
% Outputs:
%   x:    the final solution
%   fs:   a vector recording history of function values at each iteration
%   ts:   a vector recording history of step sizes used at each iteration
%   nls:  a vector recording of number of line searches at each iteration

% first we need to check the options, use default ones if not provided
if nargin < 3; opts = []; end
opts = set_options( opts );

% initialize x, y and step size
x = x0;
y = x0;
t = opts.t_fixed;

% main loop of simplified optimal method
for k = 1:opts.maxitrs;
  
    % first query the oracle
    [fy, gy] = f.oracle(y);
   
    % test stopping criterion
    if norm(gy) < opts.epsilon
        break;
    end

    % line search on negative gradient direction 
    d = - gy;
    
    % line search
    switch lower( opts.linesearch )
        case 'fixed'
            t = opts.t_fixed;
            nl = 0;
        case 'bt'
            [t, nl] = backtracking(f, y, fy, gy, d, t, opts);
        otherwise
            error('line search method not implemented');
    end
  
    % update x1 and y
    x1 = y + t*d;
    y = x1 + ((k-1)/(k+2))*(x1 - x);
    
    % record history
    fs(k) = f.oracle(x);
    ts(k) = t;
    nls(k) = nl;

    % update x for next iteration
    x = x1;
end