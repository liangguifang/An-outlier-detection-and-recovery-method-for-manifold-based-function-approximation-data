function [Ep,Index,f_approx] = MMLSforL0test(r,f,epsilon,d,m)
[~,N]=size(r);

[Alpha, Q] = PolyAlpha(d, m);
pp = zeros(N, Q);

A = zeros(N, N);
for j = 1:N
     [U,q] = FLCS(r ,r(:,j), epsilon,d);
    R_bar = r - q;
    X = U' * R_bar;
    Xn=X';
    for i = 1:N
        pp(i, :) = PnByAlpha(Xn(i, :), Alpha);
    end
    W = zeros(N, 1);
    Ik = zeros(N, 1);

    for k = 1:N
        [W(k), Ik(k)] = LW(X(:,k),0.75);
    end

    Il = find(Ik > 0);
    Wx = W(Il);
    Px = pp(Il, :);
    M = zeros(Q, Q);
    PW = Px';
    for ii = 1:Q
        PW(ii, :) = PW(ii, :) .* Wx';
        for jj = 1:Q
            M(ii, jj) = PW(ii, :) * Px(:, jj);
        end
    end

    beta=SbySVD(M,pp(j, :)');
    alpha = PW' * beta;
    A(j, Il) = alpha';
end
 f_approx = A * f;

U = diag(ones(N, 1)) - A;
R= U * f;
UN = zeros(N, 1);
for i = 1:N
    UN(i) = norm(U(:, i));
end
Um = min(UN);

e=5;

eta = Um * median(f);

Eps =0.5;

[Ep, Index] = OMPBYzspForL0(U, R,N, eta, Eps);

end
