function [U, q, converged] = FLCS(R,r,epsilon,d,h,searchRadius)
% 给定原空间搜索半径时，每次迭代都以当前 q 重新选择邻域。
if nargin < 6
    searchRadius = [];
end
q = r;
k = 4;
max_iterations = 100;

if ~isempty(searchRadius)
    offsets = bsxfun(@minus,R,r);
    initialNeighbors = sum(offsets.^2,1) < searchRadius^2;
    if ~any(initialNeighbors)
        error('FLCS:EmptyNeighborhood','No samples within the initial search radius.');
    end
    q = mean(R(:,initialNeighbors),2);
end

A_tilde = weightedOffsets(R,q,k,h,searchRadius,d);
[U, ~, ~] = svd(A_tilde,'econ');
U = U(:,1:d);
if ~isempty(searchRadius)
    q = q + U*(U.'*(r-q));
end

for iteration_count = 1:max_iterations
    q_prev = q;
    R_bar = weightedOffsets(R,q,k,h,searchRadius,d);
    X = R_bar'*U;
    X_bar = [ones(size(R_bar,2),1), X];
    alpha = SbySVD(X_bar'*X_bar,X_bar'*R_bar');
    q_bar = q + alpha(1,:)';
    [Q, ~, ~] = qr(alpha(2:end,:)',0);
    U = Q(:,1:d);
    q = q_bar+U*(U'*(r-q_bar));
    converged = norm(q-q_prev) < epsilon;
    if converged
        break;
    end
end
end

function R_bar = weightedOffsets(R,q,k,h,searchRadius,d)
R_bar = bsxfun(@minus,R,q);
distance = sqrt(sum(R_bar.^2,1));
weights = w_test(distance,k,h);
if ~isempty(searchRadius)
    active = distance < searchRadius & weights > 0;
    if nnz(active) < d+1
        error('FLCS:InsufficientNeighbors', ...
            'The current chart needs at least %d samples within the search radius.',d+1);
    end
    R_bar = R_bar(:,active);
    weights = weights(active);
end
R_bar = bsxfun(@times,R_bar,sqrt(weights));
end
