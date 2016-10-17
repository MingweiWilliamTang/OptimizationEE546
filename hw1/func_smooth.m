classdef func_smooth < handle
% An abstract class that defines interface for a differentiable function f(x).
    methods (Abstract)
        [fval, grad, Hess] = oracle(f, x);
        % 0, 1st, and 2nd order oracle (depending on nargout)

        mu = strong_convex_parameter(f);
        % return strong convexity parameter or a lower bound.
    end
end

