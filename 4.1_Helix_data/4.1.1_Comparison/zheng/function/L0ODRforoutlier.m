function [Ep, Index,f_approx] = L0ODRforoutlier(A,f,Eps)
%Copyright (c) 2025.2.25,郑三棚,梁贵方
%Refering to "Zheng, Sanpeng, Renzhong Feng, and Aitong Huang. “An Outlier Detection and Recovery Method Based on Moving Least Squares Quasi-Interpolation Scheme and ł 0 ł 0 -Minimization Problem.” 
% Applied Mathematical Modelling 122 (October 2023): 127–50. 
% https://doi.org/10.1016/j.apm.2023.05.032."

% 第二部分：保留原 L0ODR.m 中异常值识别逻辑。
% Eps 由调用方按原公式 delta^(dp-1) 计算并传入。
    n = size(A, 1);
    f_approx=A*f;
    % 正交追踪算法计算偏差值
    U = diag(ones(n, 1)) - A;%测量矩阵
    R= U * f;
    UN = zeros(n, 1);
    for i = 1:n
        UN(i) = norm(U(:, i));
    end
    Um = min(UN);
    eta = Um * abs(median(f));
    % 调用外部函数以计算偏差值
    [Ep, Index] = OMPBYzspForL0(U, R, n, eta, Eps);
end

function [c,Index,Rn] = OMPBYzspForL0(U,R,Max,eta,eps)
%OMPBYzsp zsp编写的使用OMP算法求解论文中不等式约束的L0优化问题的算法
%   返回系数c，以及c的支撑集Index
r=R;
k=0;
Max=min(Max,300);
Rn=zeros(Max+1,1);
nr=norm(r);
Rn(k+1)=nr;
Iold=[];
[~,N]=size(U);
I=1:N;
c=zeros(N,1);
NU=zeros(N,1);
for i=1:N
    NU(i)=norm(U(:,i));%求各列的模
end
while nr>eta && k<Max/4
    C=(U(:,I)'*r)./NU(I);%求每一列与r的值
    [~,ck]=max(abs(C));%取绝对值最大的一列
    Inew=[Iold,I(ck)];%将C最大值所对应的列数存入
    I(ck)=[];%将ck列从原有矩阵剔除
    At=U(:,Inew);%取出A的Inew列
    ctnew=(At'*At)\(At'*R);
    r=R-At*ctnew;%更新残差
    nr=norm(r);
    k=k+1;
    Rn(k+1)=nr;
    if k>1
        RATIO=abs(Rn(k)-Rn(k+1))/abs(Rn(k-1)-Rn(k));
        if RATIO <=eps
            break;
        end
    end
    ctold=ctnew;
    Iold=Inew;
end
   if k>0
   c(Iold)=ctold;
   Index=sort(Iold);
   else
      Index=[]; 
   end
end

