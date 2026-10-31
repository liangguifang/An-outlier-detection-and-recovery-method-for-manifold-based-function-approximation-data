function h = fillh(R,M_ref)

Nref=size(M_ref,2);

h=0;

for i=1:Nref

    diff=R-M_ref(:,i);

    dist=sqrt(sum(diff.^2,1));

    h=max(h,min(dist));

end

end
