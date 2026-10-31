function [c,Index,Rn,ratio_record,k_record] = OMPBYzspForL0(A,b,Max,eta,eps)

r=b;
k=0;
Max=Max/2;
Rn=zeros(Max+1,1);
nr=norm(r);
Rn(k+1)=nr;
Iold=[];
[~,N]=size(A);
I=1:N;
c=zeros(N,1);
NA=zeros(N,1);
for i=1:N
    NA(i)=norm(A(:,i));
end
ratio_record = [];
k_record = [];

while nr>eta && k<Max
    C=(A(:,I)'*r)./NA(I);
    [~,ck]=max(abs(C));
    Inew=[Iold,I(ck)];
    I(ck)=[];
    At=A(:,Inew);

    ctnew=At\b;
    r=b-At*ctnew;
    nr=norm(r);
    k=k+1;
    Rn(k+1)=nr;
    if k>1
        RATIO=abs(Rn(k)-Rn(k+1))/abs(Rn(k-1)-Rn(k));
        ratio_record = [ratio_record, RATIO];
        k_record = [k_record, k];
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

Rn = Rn(1:k+1);
end
