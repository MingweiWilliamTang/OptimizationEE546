function opts = set_options( opts )
% Check and set algorithmic options
%
%  Field        Default values
%------------------------------
% .epsilon      1.0e-4      stopping precision for norm of gradient
% .maxitrs      100         maximum number of iterations allowed
% .linesearch   'fixed'     line search scheme: {'fixed', 'bt'}
% .t_fixed      1.0         value for fixed step size
% .ls_alpha     0.5         backtracking (bt) line search parameter alpha
% .ls_beta      0.5         backtracking (bt) line search parameter beta
% .ls_gamma     2.0         adaptive bt line search parameter gamma
% .bt_init      't_fixed'   how to initialize backtracking line search:
%                           {'t_fixed', 'previous', 'adaptive'}
%------------------------------


if isfield(opts, 'epsilon')
    if opts.epsilon <= 0
        error('opts.epsilon should be a small positive number');
    end
else
    opts.epsilon = 1.0e-4;
end

if isfield(opts, 'maxitrs')
    if opts.maxitrs <= 0
        error('opts.maxitrs should be a positive integer');
    end
else
    opts.max_iters = 100;
end

if ~isfield(opts, 'linesearch')
    opts.linesearch = 'fixed';
end

if isfield(opts, 't_fixed')
    if opts.t_fixed <=0
        error('opts.t_fixed should be a positive number');
    end
else
    opts.t_fixed = 1.0;
end

if isfield(opts, 'ls_alpha')
    if opts.ls_alpha <=0 || opts.ls_alpha > 0.51
        error('opts.ls_alpha should be in the interval (0,0.5]');
    end
else
    opts.ls_alpha = 0.5;
end

if isfield(opts, 'ls_beta')
    if opts.ls_beta <=0 || opts.ls_beta >=1
        error('opts.ls_beta should be in the interval (0,1)');
    end
else
    opts.ls_beta = 0.5;
end

if isfield(opts, 'ls_gamma')
    if opts.ls_gamma < 1
        error('opts.ls_gamma should be no smaller than 1');
    end
else
    opts.ls_gamma = 2.0;
end

if ~isfield(opts, 'bt_init')
    opts.bt_init = 't_fixed';
end


    
