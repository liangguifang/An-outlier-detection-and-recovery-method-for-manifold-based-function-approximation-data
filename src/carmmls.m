function [Ep,Index,f_approx] = carmmls(r,f,epsilon,d,m)
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
        [W(k), Ik(k)] = LW(X(:,k),0.95);
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
R = U * f;

middle_range_len = 60;
if N <= middle_range_len
    search_idx = 1:N;
else

    start_pos = floor((N - middle_range_len) / 2) + 1;
    search_idx = start_pos : (start_pos + middle_range_len - 1);
end

U_sub = U(:, search_idx);

UN = zeros(N, 1);
for i = 1:N
    UN(i) = norm(U(:, i));
end
Um = min(UN);
eta = Um *median(f)*6;

e=4;

Eps = 0.45;

[Ep_sub, Index_sub] = OMPBYzspForL0(U_sub, R, length(search_idx), eta, Eps);

Ep = zeros(N, 1);
Ep(search_idx) = Ep_sub;

if ~isempty(Index_sub)
    Index = search_idx(Index_sub);
    Index = sort(Index);
else
    Index = [];
end

end
