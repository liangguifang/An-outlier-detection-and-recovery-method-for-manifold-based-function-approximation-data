function A = L0ODRforA(Xn,dp,delta)
%Copyright (c) 2025.2.25,郑三棚,梁贵方
%Refering to "Zheng, Sanpeng, Renzhong Feng, and Aitong Huang. “An Outlier Detection and Recovery Method Based on Moving Least Squares Quasi-Interpolation Scheme and ł 0 ł 0 -Minimization Problem.” 
% Applied Mathematical Modelling 122 (October 2023): 127–50. 
% https://doi.org/10.1016/j.apm.2023.05.032."

% 第一部分：保留原 L0ODR.m 中矩阵 A 的计算逻辑。
% Xn 为 n*d，dp 与 delta 的默认值保持原设置。
    if nargin <2
        dp = 3; % 默认值
    end
    if nargin <3
        % h=Hilldistance(Xn',[0,1;0,1]);
        % delta = 2*dp*h; % 默认值
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
        [W(k), Ik(k)] = w_test(norm(Yn(:,k)),20,delta);
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
        %beta = M \ pp(j, :)';
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
    % p=zeros(1,Q);
    x=X(:)';
    % for i=1:Q
    %     p(i)=prod(x.^Alpha(i,:));
    % end
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

function [w,k] =LW(x,delta)
%移动最小二乘法中的局部权函数，delta为影响半径，x为所求点，y为数据点
%使用Wendland的C4的CSRBF(理论之适用于d<=3的数据点)
% k表示是否在影响半径内
k1=4;
R = norm(x) ;
% if R<1
%     w=((1-R)^6)*(35*R^2+18*R+3);
%     k=1;
% else
%     w=0;
%     k=0;
% end
if (R >= -k1*delta) && (R <= k1*delta)
    w= exp(-R.^2./(R - k1*delta).^2);
    k=1;
else
    w=0;
    k=0;
end
end

function h = Hilldistance(X,Omega)
%Hilldistance 计算点集X在区域Omega内的填充距离
%   X的格式是d*N的矩阵
%   Omega是d*2的矩阵，储存每个维度的最大与最小值
%   使用DE算法近似计算
[d,N]=size(X);
%%
%初始化参数
ObjectF=@(x)(-DistanceX(x,X,N));
D=d;
IPR=Omega;
NP=10*D;
Km=min([100^d/NP,N]);
F=0.5;
CR=0.1;
%%
[Fopt,~] = DEmethod(ObjectF,D,NP,F,CR,IPR,Km);
h=-Fopt;
%%
%测试
% n=100;
% dx=(Omega(:,2)-Omega(:,1))/(n-1);
% hm=0;
% for i=1:n
%     for j=1:n
%        hij= -ObjectF(Omega(:,1)+[(i-1)*dx(1);(j-1)*dx(2)]);
%        if hij>hm
%            hm=hij;
%            xm=Omega(:,1)+[(i-1)*dx(1);(j-1)*dx(2)];
%        end
%     end
% end
end
function dis = DistanceX(x,X,N)
dis=norm(x-X(:,1));
for i=2:1:N
    disnew=norm(x-X(:,i));
    if disnew<dis
        dis=disnew;
    end
end
end
function [Fopt,xopt,Bn,time] = DEmethod(ObjectF,D,NP,F,CR,IPR,Km)
%DEmethod IPR是初始种群取值区域，它是D*2的矩阵
tic
v=zeros(D,NP);%Mutation
OldV=zeros(1,NP);%Old Value 
TrialV=zeros(1,NP);%Trial Value
OldP=zeros(D,NP);
for i=1:D
    OldP(i,:)=unifrnd (IPR(i,1), IPR(i,2), 1, NP);
end
% Km=100;
Bn=zeros(Km,1);%用于储存迭代中最优值得数组
for i=1:NP
    OldV(i)=ObjectF(OldP(:,i));
end
OB=min(OldV);
NewP=OldP;
NewV=OldV;
NB=OB;
Count=1;
Bn(Count)=NB;
while(Count<Km)
    OldP=NewP;
    OldV=NewV;
    % Mutation, Crossover and Selection
    for i=1:NP
        % Mutation
        r1=unidrnd(NP);
        while r1==i
            r1=unidrnd(NP);
        end
        r2=unidrnd(NP);
        while r2==i||r2==r1
            r2=unidrnd(NP);
        end
        r3=unidrnd(NP);
        while r3==i||r3==r1||r3==r2
            r3=unidrnd(NP);
        end
        v(:,i)=OldP(:,r1)+F*(OldP(:,r2)-OldP(:,r3));
        % Crossover
        J=unidrnd(D);
        for j=1:D
        C=unifrnd(0,1);
        if(C<CR||j==J)
            NewP(j,i)=v(j,i);
        else
            NewP(j,i)=OldP(j,i);
        end
        %%
        %现在到区域IPR内
        if NewP(j,i)<IPR(j,1)
            NewP(j,i)=IPR(j,1);
        else
            if NewP(j,i)>IPR(j,2)
             NewP(j,i)=IPR(j,2);
            end
        end
        %%
        end
        % Selection
        TrialV(i)=ObjectF(NewP(:,i));
        if TrialV(i)<OldV(i)
            NewV(i)=TrialV(i);
        else
            NewP(:,i)=OldP(:,i);
            NewV(i)=OldV(i);
        end
    end
   [NB,Index]=min(NewV);
    Count=Count+1;
    Bn(Count)=NB;
end
% 找到最优解
Fopt=NB;
xopt=NewP(:,Index);
time=toc;
end
