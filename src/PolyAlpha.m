function [A,Q] = PolyAlpha(d,m)

Q=nchoosek(d+m,d);
A=zeros(Q,d);
a=zeros(1,d);
i=2;
while i<=Q
    a=AddOne(a,m);
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
    else
        if N<=1

            error('增加溢出，请检查程序')
        else
            b=zeros(1,N);
            b(2:N)=AddOne(a(2:N),m);
        end
    end
end
end
