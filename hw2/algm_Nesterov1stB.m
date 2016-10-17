function [x, fs, ts] = algm_Nesterov1stB(f, Psi, x0, opts)
% Variant of Nesterov's 1st method for composite convex optimization 
%   minimize { F(x) = f(x) + Psi(x) }
% Inputs: 
%   f:    a function object that implements [fval, grad] = f.oracle(x)
%   Psi:  a function object that implements proximal mapping
%   x0:   the starting point for the iterative algorithm
%   opts: a struct of options and parameters for the algorithm
% Outputs:
%   x:    the final solution
%   fs:   a vector recording history of function values at each iteration
%   ts:   a vector recording history of step sizes used at each iteration

% first we need to check the options, use default ones if not provided
if nargin < 3; opts = []; end
opts = set_options( opts );


% choose to use mu provided by function or an user-specified value in opts
mu = f.strong_convex_parameter();
%mu = 0.1;


% initialize x, y and step size
x = x0;
v = x0;
t = opts.t_fixed;
tk = t;
alpha = 1;
alphak = alpha;
for k = 1:opts.maxitrs
    if k >=2 
        switch lower(opts.bt_init)
            case 't_fixed'
                t = opts.t_fixed;
            case 'previous'
                t = tk;
            case 'adaptive'
                t = min(opts.ls_gamma * tk, 1/ mu);
            otherwise
                error('method not implemented');
        end
    end
    % start line search 
    t = t / opts.ls_beta;
    for i = 0:opts.ls_maxstep
        t = t * opts.ls_beta;
        
      %  alpha = (mu * t * tk - alphak^2 * t + sqrt((mu * t * tk - alphak^2 * t)^2 + ...
       %     4 * t * tk * alphak^2)) / (2 * tk);
       a = 1 / t;
       b = -(mu - alphak^2/tk);
       c = -alphak^2/tk;
       alpha = (-b + sqrt(b^2 - 4*a*c))/(2*a);
        theta = alpha / (1 + (alpha / alphak^2)*tk * mu);
        
        y = (1 - theta) * x + theta * v;
        
        [fy,gy] = f.oracle(y);
        
        x1 = Psi.prox_mapping(y - t * gy,t);
        
        fx1 = f.oracle(x1);
        
        if fx1 <= fy + trace(gy'*(x1-y)) + norm(x1-y,'fro')^2/(2*t)
            break;
        end
    end
    v = x + (x1 -x) / alpha;
    x = x1;
    alphak = alpha;
    tk = t;    
    ts(k) = t;
    p1 = Psi.oracle(x1);
    fs(k) = fx1 + p1;
end