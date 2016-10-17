function [x,fs,ts,nls] = algm_optimal(f, x0, opts)
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

mu = f.strong_convex_parameter();
%mu = 0.1;
disp(mu);
% initialize x and step size
x = x0;
y = x;
t = opts.t_fixed;

for k = 1:opts.maxitrs;
    % first query the oracle
    [fy, gy] = f.oracle(y);
    
    % test stopping criterion
    if norm(gy) < opts.epsilon
        break;
    end

    % use negative gradient direction 
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
    
    % update x 
    x_new = y + t*d;
    q = t*mu;
    if k == 1
        a0 = 0.5;
    end
        a1 = (q - a0^2 + sqrt( (a0^2 - q)^2 + 4 * a0^2)) / 2;
        b = a0*(1-a0)/(a0^2+a1);
        
        y = x_new + b * (x_new - x);
        x = x_new;
        a0 = a1;
    % record history
    fs(k) = f.oracle(x);
    ts(k) = t;
    nls(k) = nl;
end




