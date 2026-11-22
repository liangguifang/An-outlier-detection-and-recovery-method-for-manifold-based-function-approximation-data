function [Ep,Index,f_approx] = MMLSforoutlier(A,f,U,UN,tau,eta)

N = length(f);

% A 和 U 只在这里转换一次，后续 OMP 使用稀疏矩阵。
if ~issparse(A)
    A = sparse(A);
end

if nargin < 3 || isempty(U)
    U = speye(N)-A;
elseif ~issparse(U)
    U = sparse(U);
end
if nargin < 4 || isempty(UN)
    UN = sqrt(sum(U.^2,1))';
end
f_approx = A*f;

% Residual passed to OMP.
R = U*f;

% 未指定阈值时，沿用基于观测中位数的设置。
if nargin < 6 || isempty(eta)
    eta = min(UN)*abs(median(f));
end

if nargin < 5 || isempty(tau)
    tau = 0.5;
end

% OMP异常识别
[Ep,Index] = OMPBYzspForL0(U,R,N,eta,tau);

end
