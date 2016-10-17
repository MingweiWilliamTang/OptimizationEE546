function [x, fs, ts, nls] = algm_BFGS(f, x0, opts)
if nargin < 3; opts = []; end
opts = set_options(opts);

% initialize x and step size
x = x0;
t = opts.t_fixed;

 m = length(x);
 %H_inv = eye(m);
  H = eye(m); 


% main loop of gradient method
% first query the oracle
[fx, gx] = f.oracle(x);

for k = 1:opts.maxitrs;   

    if norm(gx) < opts.epsilon
        break;
    end 

    % use negative gradient direction 
  %  d = -H_inv * gx;
   d = - H \ gx; 
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
   s = t * d;
   x = x + s;
   [fx, gx_new] = f.oracle(x);
   y = gx_new - gx;
  % H_inv = (eye(m) - (y * s')/(y'*s)) * H_inv * (eye(m) - (s * y')/(y'*s)) + (s*s')/(s'*y);
  H = H + (y * y')/(y'*s) - (H * s * s' * H) / (s' * H * s);
     gx = gx_new;
    
    % record history
    fs(k) = fx;
    ts(k) = t;
    nls(k) = nl;
    
    % test stopping criterion
  %  if norm(gx) < opts.epsilon
  %      break;
  %  end
    
end




