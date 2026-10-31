function c = SbySVD(A,b)

[U,S,V] = svd(A);
[l,h]=size(S);
iS=zeros(l,h);
for i=1:min([l,h])
    if S(i,i)>0
    iS(i,i)=1/S(i,i);
    end
end
c=(V*iS'*U')*b;
end
