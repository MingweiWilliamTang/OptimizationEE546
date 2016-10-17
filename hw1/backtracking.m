function [t, nl] = backtracking(f, x, fx, gx, d, t_pre, opts)
% backtracking line search to find t>0 such that
%   f(x+t*d) <= f(x) + t*alpha*(d'*gx)
% only 0-order oracle for f is needed here (no gradients or Hessian)
% Inputs: 
%   f:  function object that implements method oracle(x)
%   x:  the current point
%   fx: value of f at x
%   gx: gradient at x
%   d:  search direction
%   t_pre: previous stepsize used
%   opts: algorithmic options (see set_options.m)
% Outputs: 
%   t:  step size choosen by line search
%   nl: number of line search iterations

% choose initial stepsize for backtracking line search
switch lower( opts.bt_init )
    case 't_fixed'
        t = opts.t_fixed;
    case 'previous'
        t = t_pre;
    case 'adaptive'
        t = t_pre*opts.ls_gamma;
    otherwise
        error('Unknown initialization for backtracking line search');
end

% initialize number of backtrackings in line search
% main loop of backtracking line search
nl = 0;
x_new = x + t * d;
fx_new = f.oracle(x_new);
delta = fx_new - fx - (opts.ls_alpha * t) * d' * gx;

while(fx_new > (fx + opts.ls_alpha * t * d' * gx) )
    t = t * opts.ls_beta;
    x_new = x + t*d;
    fx_new = f.oracle(x_new);
    nl = nl + 1;
    if nl > 1000
        break;
    end
end

end
    
