function [x, fs, ts, nls] = algm_gradient(f, x0, opts)
% the gradient method for smooth convex optimization (unconstrained)
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

% initialize x and step size
x = x0;
t = opts.t_fixed;

% main loop of gradient method
for k = 1:opts.maxitrs;
    % first query the oracle
    [fx, gx] = f.oracle(x);
    
    % test stopping criterion
    if norm(gx) < opts.epsilon
        break;
    end

    % use negative gradient direction 
    d = - gx;
    
    % line search
    switch lower( opts.linesearch )
        case 'fixed'
            t = opts.t_fixed;
            nl = 0;
        case 'bt'
            [t, nl] = backtracking(f, x, fx, gx, d, t, opts);
        otherwise
            error('line search method not implemented');
    end
    
    % update x 
    x = x + t*d;
    
    % record history
    fs(k) = fx;
    ts(k) = t;
    nls(k) = nl;
end




