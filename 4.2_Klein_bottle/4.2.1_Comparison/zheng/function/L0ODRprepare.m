function P = L0ODRprepare(A)
% Precompute fixed quantities for 100 repeated L0ODR trials.
% Input: A is n-by-n, finite, real, fixed across trials.
% e=0.6 exactly matches the uploaded L0ODRforoutlier.m.

if ~isnumeric(A) || ~isreal(A) || ndims(A) ~= 2 || ...
        size(A,1) ~= size(A,2) || isempty(A) || any(~isfinite(A(:)))
    error('A must be a finite real square matrix.');
end

n = size(A,1);
P.A = A;
P.U = eye(n) - A;
P.normU = sqrt(sum(P.U.^2,1)).';
P.eta = 0.6 * min(P.normU);
end