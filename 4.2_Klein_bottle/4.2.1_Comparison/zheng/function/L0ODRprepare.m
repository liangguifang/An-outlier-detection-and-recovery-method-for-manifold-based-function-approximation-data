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
if issparse(A)
    P.U = speye(n) - A;
    % 当非零元较多时，完整矩阵乘向量通常更快。
    if nnz(P.U) > 0.20*n^2
        P.U = full(P.U);
    end
else
    P.U = eye(n) - A;
end
P.normU = full(sqrt(sum(P.U.^2,1)).');
P.eta = 0.6 * min(P.normU);
end