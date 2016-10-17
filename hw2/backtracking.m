function [t, x1, Gx] = backtracking(f, Psi, x, fx, gx, t_pre, opts)
% line search to find t>0 such that x1 = prox_Psi(x-t*gx,t) satisfies
%   f(x1) <= f(x) + gx'*(x1-x) + (1/2*t)*||x1 - x||^2 (see page 7-18)
% Inputs: 
%   f:  function object that implements method oracle(x)
%   Psi:simple function that implements prox_mapping(z, t)
%   x:  the current point
%   fx: value of f at x
%   gx: gradient at x
%   t_pre: previous stepsize used
%   opts: algorithmic options (see set_options.m)
% Outputs: 
%   t:  step size choosen by line search
%   x1: prox_mapping of x with step size t
%   Gx: the gradient mapping at x

% choose initial stepsize for backtracking line search
switch lower( opts.bt_init )
    case 't_fixed'
        t = opts.t_fixed;
    case 'previous'
        t = t_pre;
    case 'adaptive'
        t = min(t_pre*opts.ls_gamma, opts.ls_maxstep);
    otherwise
        error('Unknown initialization for backtracking line search');
end

% line search loop
x1 = Psi.prox_mapping(x - t * gx, t);
f_new = f.oracle(x1);
dx = x1 - x;
%i=1;
while(f_new > fx + trace(gx' * dx) + norm(dx,'fro')^2 / (2*t))
    t = t * opts.ls_beta;
    x1 = Psi.prox_mapping(x - t * gx, t);
    f_new = f.oracle(x1);
    dx = x1 - x;
   % if i > opts.maxitrs
   %     break
   % else
   %     i = i+1;
   % end 
end
Gx = -dx / t;
end     % end of function backtracking()
