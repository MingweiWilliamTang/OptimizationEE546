classdef func_logistic_l2 < func_smooth
% objective function: logistic regression with l2 regularization
% f(x) = (1/m) \sum_{i=1}^m log(1 + exp(-a_i'*x)) + (lambda/2)*||x||_2^2
    properties
        A       % an m by n real matrix
        lambda  % l2 regularization parameter
    end
    
    methods
        function f = func_logistic_l2(A, lambda)
        % constructor for logistic regression
            f.A = A;
            f.lambda = lambda;
        end
        
        function [fval, grad] = oracle(f, x)
        % 0 and 1st order oracle (depending on nargout)
        
            % compute function value
            [m, n] = size(f.A);
            Ax = f.A*x;
            e_Ax = exp(-Ax);
            % fval = (1.0/m)*sum(log(1+e_Ax)) + (f.lambda/2)*(x'*x);
            % the implementation below is much more robust
            fval = (1.0/m)*(sum(log(1+e_Ax(Ax>-50))) - sum(Ax(Ax<=-50))) ...
                + (f.lambda/2)*(x'*x);
            if nargout <= 1; return; end
                
            % compute gradient vector
            p = 1./(1+e_Ax);
            grad = (-1.0/m)*(f.A'*(1-p)) + f.lambda*x;
        end
     
        function mu = strong_convex_parameter(f)
        % return a lower bound on strong convexity parameter
            mu = f.lambda;
        end
    end
end
    