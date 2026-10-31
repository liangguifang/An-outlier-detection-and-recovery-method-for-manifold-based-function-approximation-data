function [w,k] =LW(x,delta)

R = norm(x) ;

if (R >= -delta) && (R <= delta)
    w= exp(-R.^2./(R - delta).^2);
    k=1;
else
    w=0;
    k=0;
end

end
