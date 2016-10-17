classdef func_nuclearnorm < func_simple
% the weighted nuclear norm: Psi(x) = lambda*||X||_*
    properties
        lambda  % weight for nuclear norm regularization
        mu      % strong convexity parameter 
    end
    
    methods
        function Psi = func_nuclearnorm(lambda)
        % construct the weighted nuclear norm function
            Psi.lambda = lambda;
            Psi.mu = 0;
        end

        function [fval, subg] = oracle(Psi, X)
        % Return function value Psi(X)
            s = svd(X);
            fval = Psi.lambda*sum(s);
            if nargout <= 1; return; end;
            
            % computing subgradient is not implemented
            error('subgradient of nuclear norm is not implemented');
        end
        
        function  X = prox_mapping(Psi, Z, t)
        % Return: argmin_X { (1/2)||X-Z||_F^2 + t*lambda*||X||_* }
        % same as argmin_X { (1/2*t)||X-Z||_F^2 + lambda*||X||_* }
    
        [U,Sigma,V] = svd(Z);
       % [r, c] = size(Sigma);
        %di = 1:(r+1):(r*c);
        Sigma = max(Sigma-t*Psi.lambda,0);
        X = U * Sigma * V';
        end
        
        function mu = strong_convex_parameter(R)
        % Return (strong) convexity parameter
            mu = R.mu;
        end
    end
end
            
       