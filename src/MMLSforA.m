function A = MMLSforA(r,epsilon,d,m,h,delta)

[~,N] = size(r);

[Alpha,Q] = PolyAlpha(d,m);
pp = zeros(N,Q);

A = zeros(N,N);

for j = 1:N

    [U,q] = FLCS(r,r(:,j),epsilon,d,h);

    R_bar = r-q;
    X = U'*R_bar;
    Xn = X';

    for i = 1:N
        pp(i,:) = PnByAlpha(Xn(i,:),Alpha);
    end

    W = zeros(N,1);
    Ik = zeros(N,1);
    distance = sqrt(sum((r-q).^2,1));

    for k = 1:N
         [W(k),Ik(k)] = LW(distance(k),delta);
    end

    Il = find(Ik>0);
    length(Il);
    if length(Il) < Q
    fprintf('j=%d, 邻域点=%d, Q=%d\n', ...
        j,length(Il),Q);
end

    Wx = W(Il);
    Px = pp(Il,:);

    PW = Px';

    for ii = 1:Q
        PW(ii,:) = PW(ii,:).*Wx';
    end

    M = PW*Px;

    beta = SbySVD(M,pp(j,:)');

    alpha = PW'*beta;

    A(j,Il) = alpha';

end

end
