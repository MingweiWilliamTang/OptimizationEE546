function [x, fs, ts, nls] = algm_Newton(f, x0, opts)

if nargin < 3; opts = []; end
opts = set_options(opts);

x = x0;
t = opts.t_fixed;


% main loop of gradient method
for k = 1:opts.maxitrs;
    % first query the oracle
    [fx, gx, hx] = f.oracle1(x);
    
    % test stopping criterion
    if norm(gx) < opts.epsilon
        break;
    end

    % gradient direction normalized by Hessian
    d = - hx \ gx;
    
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
    x = x + t * d;
    
    % record history
    fs(k) = fx;
    ts(k) = t;
    nls(k) = nl;
end