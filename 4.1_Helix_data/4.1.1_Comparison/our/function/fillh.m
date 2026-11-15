function h = fillh(R,M_ref)

% R: 采样点
% M_ref: 流形稠密参考点

Nref=size(M_ref,2);

h=0;

for i=1:Nref

    diff=R-M_ref(:,i);

    dist=sqrt(sum(diff.^2,1));

    h=max(h,min(dist));

end

end
