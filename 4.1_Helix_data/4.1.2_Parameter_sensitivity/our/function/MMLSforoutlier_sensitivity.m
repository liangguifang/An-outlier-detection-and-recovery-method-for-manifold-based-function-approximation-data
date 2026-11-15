function [Ep,Index,f_approx] = MMLSforoutlier_sensitivity(A,f,etaFactor,Eps,U,Um)
% Parameterized outlier detector for the sensitivity study.
if nargin < 3 || isempty(etaFactor), etaFactor = 0.6; end
if nargin < 4 || isempty(Eps), Eps = 0.05; end
f = f(:);
N = numel(f);
if ~isequal(size(A),[N,N])
    error('MMLSforoutlier_sensitivity:SizeMismatch','A must be N-by-N.');
end
if nargin < 5 || isempty(U), U = speye(N)-sparse(A); else, U = sparse(U); end
if nargin < 6 || isempty(Um), Um = min(sqrt(sum(U.^2,1))); end
if ~isscalar(etaFactor) || ~isfinite(etaFactor) || etaFactor < 0
    error('MMLSforoutlier_sensitivity:BadEta','etaFactor must be nonnegative.');
end
if ~isscalar(Eps) || ~isfinite(Eps) || Eps < 0
    error('MMLSforoutlier_sensitivity:BadEps','Eps must be nonnegative.');
end

A = sparse(A);
f_approx = A*f;
R = U*f;
eta = etaFactor*Um;
[Ep,Index] = OMPBYzspForL0(U,R,N,eta,Eps);
Ep = Ep(:);
end

