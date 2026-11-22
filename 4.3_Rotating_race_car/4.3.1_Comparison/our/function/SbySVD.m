function c = SbySVD(A,b)
%SbySVD 使用SVD求解c=A\b;
[U,S,V] = svd(A); % A=U*S*V';
[l,h]=size(S);
iS=zeros(l,h);
for i=1:min([l,h])
    if S(i,i)>0
    iS(i,i)=1/S(i,i);
    end
end
c=(V*iS'*U')*b;
end