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
        % 0, 1st, and 2nd order oracle (depending on nargout)

        % TO BE COMPLETED BELOW
        % You can mimic implementation of oracle() in func_logsumexp.m
        [m,n] = size(f.A);
        %nAx = -f.A*x;
        %mnxAx = max(nAx)
        expAx = exp(-f.A * x);
        fval = sum(log(1 + expAx))/m + f.lambda * x' * x / 2;
        %fval = sum(nAx + log(1./ eAx)) / m + lambda * x' * x / 2;
        %grad = lambda * x - A' * ((eAx ./ (1 + eAx)) / 2
        grad = -f.A' * (expAx./(1 + expAx)) / m + f.lambda * x;
        %Hess = f.A' * diag((exp(-f.A * x)) ./(1 + exp(-f.A * x))^.2) * f.A / m +  f.lambda * eye(n);
        end
        function [fval, grad, Hess] = oracle1(f, x)
        % 0, 1st, and 2nd order oracle (depending on nargout)

        % TO BE COMPLETED BELOW
        % You can mimic implementation of oracle() in func_logsumexp.m
        [m,n] = size(f.A);
        %nAx = -f.A*x;
        %mnxAx = max(nAx)
        expAx = exp(-f.A * x);
        fval = sum(log(1 + expAx))/m + f.lambda * x' * x / 2;
        %fval = sum(nAx + log(1./ eAx)) / m + lambda * x' * x / 2;
        %grad = lambda * x - A' * ((eAx ./ (1 + eAx)) / 2
        grad = -f.A' * (expAx./(1 + expAx)) / m + f.lambda * x;
        Hess = f.A' * diag(expAx ./(1 + expAx).^2) * f.A / m +  f.lambda * eye(n);
        end
        
        
        function mu = strong_convex_parameter(f)
        % return a lower bound on strong convexity parameter
            mu = f.lambda;
        end
    end
end
    