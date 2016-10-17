classdef func_logsumexp < func_smooth
% the log-sum-exp functin: f(x) = log( \sum_{i=1}^m exp(a_i'*x + b_i) )
    properties
        A       % an m by n matrix, each row is a_i transpose 
        b       % an m by 1 vector
    end
    
    methods
        function f = func_logsumexp(A, b)
        % need A and b to construct the function
            f.A = A;
            f.b = b;
        end
        
        function [fval, grad] = oracle(f, x)
        % 0, 1st, and 2nd order oracle (depending on nargout)
            
            % compute function value
            eAxb = exp(f.A*x+f.b);
            fval = log(sum(eAxb));
            if nargout <= 1; return; end
                
            % compute gradient vector
            s = sum(eAxb);
            grad = (1.0/s)*(f.A'*eAxb);
            if nargout <= 2; return; end
            
            % compute Hessian matrix
            %Hess = (1.0/s)*f.A'*diag(eAxb)*f.A - (1.0/s^2)*(grad*grad');
        end
         function [fval, grad, Hess] = oracle1(f, x)
        % 0, 1st, and 2nd order oracle (depending on nargout)
            
            % compute function value
            eAxb = exp(f.A*x+f.b);
            fval = log(sum(eAxb));
            if nargout <= 1; return; end
                
            % compute gradient vector
            s = sum(eAxb);
            grad = (1.0/s)*(f.A'*eAxb);
            if nargout <= 2; return; end
            
            % compute Hessian matrix
            Hess = (1.0/s)*f.A'*diag(eAxb)*f.A - (1.0/s^2)*(grad*grad');
        end
        function mu = strong_convex_parameter(f)
        % return a lower bound on strong convex parameter
            mu = 0;
        end
    end
end
    