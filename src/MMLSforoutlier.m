function [Ep,Index,f_approx] = MMLSforoutlier(A,f)

N = length(f);

f_approx = A*f;

U = eye(N)-A;

R = U*f;

UN = zeros(N,1);

for i = 1:N
    UN(i) = norm(U(:,i));
end

Um = min(UN);

eta = Um*abs(median(f));

Eps = 0.55;

[Ep,Index] = OMPBYzspForL0(U,R,N,eta,Eps);

end
