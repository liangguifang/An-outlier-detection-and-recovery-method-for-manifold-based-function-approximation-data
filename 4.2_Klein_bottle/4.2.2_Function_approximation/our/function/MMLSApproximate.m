function [values,Btest] = MMLSApproximate(rTrain,rTest,fTrain,epsilon,d,m,h,delta)
% Evaluate the MLS approximation at independent test points.
% The MLS matrix A is built only on rTrain. This routine constructs the
% cross-evaluation weights B_test (Ntest-by-N) and never fits a matrix on
% rTest or uses test values for outlier detection.

[~,N] = size(rTrain);
[~,Ntest] = size(rTest);
[Alpha,Q] = PolyAlpha(d,m);
values = zeros(Ntest,1);
Btest = zeros(Ntest,N);

for j = 1:Ntest
    [U,q] = FLCS(rTrain,rTest(:,j),epsilon,d,h);
    X = U'*(rTrain-q);
    x0 = U'*(rTest(:,j)-q);

    % FLCS projects the evaluation point to the local affine origin.
    % Use the same projected-distance convention as MMLSforA.m.
    distance = sqrt(sum(X.^2,1));
    [W,Ik] = w_test(distance,delta/h,h);
    Il = find(Ik);
    if numel(Il) < Q
        error('MMLSApproximate:NotEnoughPoints', ...
            'Test point %d has %d neighbors but Q=%d (min distance %.6g, delta %.6g).', ...
            j,numel(Il),Q,min(distance),delta);
    end

    Px = PnByAlpha(X(:,Il)',Alpha);
    b = PnByAlpha(x0',Alpha);
    Wx = W(Il);
    PW = bsxfun(@times,Px',Wx);
    M = PW*Px;
    if any(~isfinite(M(:)))
        error('MMLSApproximate:NonfiniteMatrix', ...
            'Test point %d produced a nonfinite local MLS matrix.',j);
    end

    beta = SbySVD(M,b');
    alpha = PW'*beta;
    Btest(j,Il) = alpha';
end
values = Btest*fTrain;
end

