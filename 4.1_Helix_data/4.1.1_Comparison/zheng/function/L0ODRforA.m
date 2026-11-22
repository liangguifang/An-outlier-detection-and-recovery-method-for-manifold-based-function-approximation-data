function A = L0ODRforA(Xn,dp,delta)
%Copyright (c) 2025.2.25,郑三棚,梁贵方
%Refering to "Zheng, Sanpeng, Renzhong Feng, and Aitong Huang. “An Outlier Detection and Recovery Method Based on Moving Least Squares Quasi-Interpolation Scheme and ł 0 ł 0 -Minimization Problem.” 
% Applied Mathematical Modelling 122 (October 2023): 127–50. 
% https://doi.org/10.1016/j.apm.2023.05.032."

% Construct the Zheng MLS matrix from the sampled coordinates.
    if nargin <2
        dp = 3; % 默认值
    end

    % 初始化
    n = size(Xn, 1); % 样本数量
    d = size(Xn, 2); % 维数

    % 计算基函数的数值
    [Alpha, Q] = PolyAlpha(d, dp); % 使用的多项式基底指标
    pp = zeros(n, Q);
    %计算基底值
    for i = 1:n
        pp(i, :) = PnByAlpha(Xn(i, :), Alpha);
    end
    
    % 计算MLS下的矩阵A
    A = zeros(n, n);
    for j = 1:n
        W = zeros(n, 1);
        Ik = zeros(n, 1);
        %计算权重
        Yn=Xn';
        for k = 1:n
        % 以当前查询点 j 为中心，计算样本点 k 的局部权重。
        [W(k), Ik(k)] = w_test(norm(Yn(:,k)-Yn(:,j)),20,delta);
        end
        Il = find(Ik > 0);
        Wx = W(Il);
        Px = pp(Il, :);
        M = zeros(Q, Q);
        PW = Px';
        for ii = 1:Q
            PW(ii, :) = PW(ii, :) .* Wx';
            for jj = 1:Q
                M(ii, jj) = PW(ii, :) * Px(:, jj);
            end
        end
        beta=SbySVD(M,pp(j, :)');
        alpha = PW' * beta;
        A(j, Il) = alpha';
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
