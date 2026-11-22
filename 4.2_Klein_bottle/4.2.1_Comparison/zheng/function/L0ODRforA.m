function [A,supportRadius] = L0ODRforA(Xn,dp,delta,fastAssembly)
%Copyright (c) 2025.2.25,郑三棚,梁贵方
%Refering to "Zheng, Sanpeng, Renzhong Feng, and Aitong Huang. “An Outlier Detection and Recovery Method Based on Moving Least Squares Quasi-Interpolation Scheme and ł 0 ł 0 -Minimization Problem.” 
% Applied Mathematical Modelling 122 (October 2023): 127–50. 
% https://doi.org/10.1016/j.apm.2023.05.032."

% Construct the Zheng MLS matrix from the sampled coordinates.
    if nargin <2
        dp = 3; % 默认值
    end

    supportFactor = 4;
    supportRadius = supportFactor*delta;
    if nargin < 4
        fastAssembly = false;
    end

    [n,d] = size(Xn);
    [Alpha,Q] = PolyAlpha(d,dp);
    pp = zeros(n,Q);
    for i = 1:n
        pp(i,:) = PnByAlpha(Xn(i,:),Alpha);
    end

    Yn = Xn.';
    if fastAssembly
        pairWeights = symmetricWeights(Yn,supportFactor,delta,supportRadius);
        columns = cell(n,1);
        coefficients = cell(n,1);
        neighborCount = zeros(n,1);
    else
        A = zeros(n,n);
    end

    for j = 1:n
        if fastAssembly
            W = pairWeights(:,j);
            Il = find(~isnan(W));
        else
            W = zeros(n,1);
            Ik = false(n,1);
            for k = 1:n
                [W(k),Ik(k)] = w_test(norm(Yn(:,k)-Yn(:,j)),supportFactor,delta);
            end
            Il = find(Ik);
        end

        Wx = W(Il);
        Px = pp(Il,:);
        M = zeros(Q,Q);
        PW = Px';
        % 三次基在克莱因瓶上存在相关性，保留原法方程的浮点计算顺序。
        for ii = 1:Q
            PW(ii,:) = PW(ii,:).*Wx';
            for jj = 1:Q
                M(ii,jj) = PW(ii,:)*Px(:,jj);
            end
        end
        beta = SbySVD(M,pp(j,:)');
        alpha = PW'*beta;
        if fastAssembly
            columns{j} = Il;
            coefficients{j} = alpha;
            neighborCount(j) = numel(Il);
        else
            A(j,Il) = alpha';
        end
    end

    if fastAssembly
        rows = repelem((1:n).',neighborCount);
        A = sparse(rows,vertcat(columns{:}),vertcat(coefficients{:}),n,n);
    end
end

function weights = symmetricWeights(X,supportFactor,h,supportRadius)
n = size(X,2);
% NaN 标识支撑域外的点，与支撑域内下溢为零的权重区分。
weights = nan(n,n);
screenRadius = supportRadius + 8*eps(supportRadius);
for j = 1:n
    offsets = bsxfun(@minus,X(:,j:n),X(:,j));
    % 无穷范数给出欧氏距离的下界，候选点仍用原欧氏范数判断。
    candidate = find(max(abs(offsets),[],1) <= screenRadius);
    for t = candidate
        k = j+t-1;
        [weight,inside] = w_test(norm(offsets(:,t)),supportFactor,h);
        if inside
            weights(k,j) = weight;
            weights(j,k) = weight;
        end
    end
end
end
function p=PnByAlpha(X,Alpha)
%PnByAlpha 在多指标集Alpha下多项式基底在X处的取值
%输出为行向量
[Q,d]=size(Alpha);
if length(X)~=d
    error('输入X错误')
else
    x=X(:)';
     p= prod(x.^Alpha, 2)';
end
end

function [A,Q] = PolyAlpha(d,m)
%PolyAlpha 根据输入返回总元次数不高于m的d元多项式的多指标集合
%   输入：d与m表示元数与次数
%   输出：A表示指标的集合，Q表示指标的个数，A是Q*d的
Q=nchoosek(d+m,d);
A=zeros(Q,d);
a=zeros(1,d);
i=2;
while i<=Q
    a=AddOne(a,m);%以m+1进制的形式使a增加1
    if sum(a)<=m
        A(i,:)=a;
        i=i+1;
    end
end
end

function b=AddOne(a,m)
N=length(a);
H=a(1);
if H>m
    error('不符合m+1近制，请检查程序')
else
    Ha=H+1;
    if Ha<m+1
        b=a;
        b(1)=Ha;
    else%需要近一位
        if N<=1
            %%不能增加
            error('增加溢出，请检查程序')
        else
            b=zeros(1,N);
            b(2:N)=AddOne(a(2:N),m);
        end
    end
end
end
