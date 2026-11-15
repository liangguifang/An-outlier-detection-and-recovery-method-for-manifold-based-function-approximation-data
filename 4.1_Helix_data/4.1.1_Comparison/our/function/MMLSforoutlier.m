function [Ep,Index,f_approx] = MMLSforoutlier(A,f,U,UN)

N = length(f);

% A 和 U 只在这里转换一次，后续 OMP 使用稀疏矩阵。
if ~issparse(A)
    A = sparse(A);
end

% MMLS拟合结果
if nargin < 3 || isempty(U)
    U = speye(N)-A;
elseif ~issparse(U)
    U = sparse(U);
end
if nargin < 4 || isempty(UN)
    UN = sqrt(sum(U.^2,1))';
end
if ~issparse(U)
    U = sparse(U);
end
f_approx = A*f;

% 构造论文中的 U = I-A
% 构造残差
R = U*f;

% 计算每一列的模
Um = min(UN);

% 保留你原程序的参数
%eta = Um;
eta =Um*abs(median(f));
%lxx0.55
Eps = 0.5;

% OMP异常识别
[Ep,Index] = OMPBYzspForL0(U,R,N,eta,Eps);

end


