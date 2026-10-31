function [f,Ik] = w_test(r,k,h)

R = k*h;

Ik = (r >= 0) & (r < R);

f = zeros(size(r));

f(Ik) = exp(-r(Ik).^2 ./ (r(Ik)-R).^2);

end
