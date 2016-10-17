function [x, Fs, ts] = algm_proxgrad(f, Psi, x0, opts)
% proximal gradient method for composite convex optimization 
%   minimize { F(x) = f(x) + Psi(x) }
% Inputs: 
%   f:    a function object that implements [fval, grad] = f.oracle(x)
%   Psi:  a function object that implements proximal mapping
%   x0:   the starting point for the iterative algorithm
%   opts: a struct of options and parameters for the algorithm
% Outputs:
%   x:    the final solution
%   Fs:   a vector recording history of function values at each iteration
%   ts:   a vector recording history of step sizes used at each iteration

% first we need to check the options, use default ones if not provided
if nargin < 3; opts = []; end
opts = set_options( opts );

% initialize x and step size
x = x0;
t = opts.t_fixed;

% main loop of proximal gradient method
for k = 1:opts.maxitrs;

    % first query the oracle
    [fx, gx] = f.oracle(x);
    
    Px = Psi.oracle(x);
    Fx = fx + Px;
    % line search
    switch lower( opts.linesearch )
        case 'fixed'
            t = opts.t_fixed;
            % apply proximal mapping of Psi and compute gradient mapping
            x1 = Psi.prox_mapping(x - t*gx, t);
            % compute gradient mapping
            Gx = (x - x1)/t;
        case 'bt'
            [t, x1, Gx] = backtracking(f, Psi, x, fx, gx, t, opts);
        otherwise
            error('line search method not implemented');
    end
    
    % record history
    Fs(k) = Fx;
    ts(k) = t;

    % stopping criterion: stop if norm of gradient mapping is small
    if norm(Gx,'fro') < opts.epsilon
        break;
    end
    
    % update x and continute to next iteration
    x = x1;

end




