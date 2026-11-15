function p = PnByAlpha(X,Alpha)
% PnByAlpha 计算多项式基。X 可为 1xd 或 nxd，Alpha 为 Qxd。

[~,d] = size(Alpha);
if size(X,2) ~= d
    error('X 的列数必须与 Alpha 的列数一致。');
end

if isvector(X)
    x = X(:)';
    p = prod(x.^Alpha,2)';
    return;
end

n = size(X,1);
Q = size(Alpha,1);
p = ones(n,Q);

for ell = 1:d
    p = p .* bsxfun(@power,X(:,ell),Alpha(:,ell)');
end
end
