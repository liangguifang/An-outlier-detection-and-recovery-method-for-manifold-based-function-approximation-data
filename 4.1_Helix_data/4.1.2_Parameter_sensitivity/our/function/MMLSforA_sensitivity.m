function [A, neighborCount, minRcond] = MMLSforA_sensitivity(r,epsilon,d,m,h,kMLS)
% Parameterized MMLS matrix builder used by LXX100_sensitivity.
% The local projection, projected distance, basis, and weight all match
% MMLSforA.m. kMLS is the coefficient in delta = kMLS*h.

if ~isscalar(kMLS) || ~isfinite(kMLS) || kMLS <= 0
    error('MMLSforA_sensitivity:BadK','kMLS must be positive.');
end
if ~isscalar(h) || ~isfinite(h) || h <= 0
    error('MMLSforA_sensitivity:BadH','h must be positive.');
end

[~,N] = size(r);
[Alpha,Q] = PolyAlpha(d,m);
A = zeros(N,N);
neighborCount = zeros(N,1);
minRcond = inf;

for j = 1:N
    [U,q] = FLCS(r,r(:,j),epsilon,d,h);
    X = U'*(r-q);
    Xn = X';

    pp = ones(N,Q);
    for ell = 1:d
        pp = pp .* bsxfun(@power,Xn(:,ell),Alpha(:,ell)');
    end

    distance = sqrt(sum(X.^2,1))';
    [W,Ik] = w_test(distance,kMLS,h);
    Il = find(Ik);
    neighborCount(j) = numel(Il);

    if numel(Il) < Q
        error('MMLSforA_sensitivity:NotEnoughPoints', ...
            'Row %d has %d neighbors but Q=%d.',j,numel(Il),Q);
    end

    Px = pp(Il,:);
    Wx = W(Il);
    PW = bsxfun(@times,Px',Wx');
    M = PW*Px;
    if any(~isfinite(M(:))) || rank(M) < Q
        error('MMLSforA_sensitivity:RankDeficient', ...
            'Row %d has a rank-deficient local MLS matrix.',j);
    end

    minRcond = min(minRcond,rcond(M));
    beta = SbySVD(M,pp(j,:)');
    alpha = PW'*beta;
    if any(~isfinite(alpha))
        error('MMLSforA_sensitivity:NonfiniteWeights', ...
            'Row %d produced nonfinite MLS weights.',j);
    end
    A(j,Il) = alpha';
end
end

