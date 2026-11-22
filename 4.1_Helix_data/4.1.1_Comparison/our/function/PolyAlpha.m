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