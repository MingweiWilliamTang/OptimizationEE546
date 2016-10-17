function [x, Fs, ts] = algm_subgradient(f, Psi, x0, opts)
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

% initialization
x = x0;
t = opts.t_fixed;

% main loop of the algorithm
for k = 1:opts.maxitrs
    [fx,gx] =f.oracle(x);
    [Px,gPx] = Psi.oracle(x);
    Fx = fx + Px;
    Gx = gx + gPx;
    g = norm(Gx);
    switch lower(opts.subg_stepsize)
        case 't_const'
            x1 = x - t * Gx;
            ts(k) = t;
        case 't_harmonic'
            x1 = x - t * Gx / k;
            ts(k) = t / k;
        case 't_sqrt'
            x1 = x - t * Gx / sqrt(k);
            ts(k) = t / sqrt(k);
        case 's_const'
            x1 = x - t * Gx / g;
            ts(k) = t / g;
        case 's_harmonic'
            x1 = x - t * Gx / g / k;
            ts(k) = t / g / k;
        case 's_sqrt'
            x1 = x - t * Gx / g / sqrt(k);
            ts(k) = t /g / sqrt(k);
    end
    Fs(k) = Fx;
    
     if norm(Gx,'fro') < opts.epsilon
        break;
     end
    
     x = x1;
 end