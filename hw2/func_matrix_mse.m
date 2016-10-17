classdef func_matrix_mse < func_smooth
% objective function: mean-squared-error of matrix completion 
% f(X) = (1/|O|)\sum_{(i,j)\in O} (X_{ij} - S_{ij})^2
    properties
        S       % sparse matrix storing observed scores
    end
    
    methods
        function f = func_matrix_mse(S)
        % constructor for matrix mean-squared-error
            f.S = S;
        end
        
        function [fval, grad] = oracle(f, X)
        % 0 and 1st order oracle (depending on nargout)
        [i,j,c] = find(f.S);
        d = length(c);
        linIdx = sub2ind(size(f.S),i,j);
        fval = norm(c - X(linIdx),'fro')^2 / d;
        grad = full(sparse(i,j,2*( X(linIdx) - c)/d));
        end
     
        function mu = strong_convex_parameter(f)
        % return a lower bound on strong convexity parameter
            mu = 0;
            %mu = 1/length(f.s);
        end
    end
end
    