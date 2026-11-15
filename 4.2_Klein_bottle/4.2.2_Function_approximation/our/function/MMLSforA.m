function [A, geometry] = MMLSforA(r,epsilon,d,m,h,delta)

[~,N] = size(r);

[Alpha,Q] = PolyAlpha(d,m);
pp = zeros(N,Q);

A = zeros(N,N);
Dproj = zeros(N,N);

for j = 1:N

    % 1. 计算第 j 个点的局部仿射平面
    [U,q] = FLCS(r,r(:,j),epsilon,d,h);

    % 2. 投影到局部仿射平面
    R_bar = r-q;
    X = U'*R_bar;
    Xn = X';
    Dproj(j,:) = sqrt(sum((X-X(:,j)).^2,1));

    % 3. 向量化计算全部点的多项式基，结果为 N x Q
    pp = ones(N,Q);
    for ell = 1:d
        pp = pp .* bsxfun(@power, Xn(:,ell), Alpha(:,ell)');
    end

    % 4. MLS权重
    distance = sqrt(sum(X.^2,1));
    Ik = (distance >= 0) & (distance < delta);
    W = zeros(N,1);
    W(Ik) = exp(-distance(Ik).^2 ./ (distance(Ik)-delta).^2);

    Il = find(Ik>0);
    length(Il);
    if length(Il) < Q
    fprintf('j=%d, 邻域点=%d, Q=%d\n', ...
        j,length(Il),Q);
end

    Wx = W(Il);
    Px = pp(Il,:);

    % 5. 构造 P'WP
    PW = Px';

    for ii = 1:Q
        PW(ii,:) = PW(ii,:).*Wx';
    end

    M = PW*Px;

    % 6. 计算 shape function
    beta = SbySVD(M,pp(j,:)');

    alpha = PW'*beta;

    A(j,Il) = alpha';

end

omega_mask = min(Dproj,Dproj') < delta;
if any(omega_mask,'all')
    DprojT = Dproj';
    omega_delta = max(abs(Dproj(omega_mask)-DprojT(omega_mask)));
else
    omega_delta = 0;
end
geometry.Dproj = Dproj;
geometry.omega_delta = omega_delta;

end



