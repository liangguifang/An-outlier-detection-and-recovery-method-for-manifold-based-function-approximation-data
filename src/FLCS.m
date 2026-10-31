function [U, q] = FLCS(R, r, epsilon, d,h)
[~, N] = size(R);

[U, ~, ~] = svds(R, d);
q = r;
q_prev = q;

k=4;
max_iterations =100;
iteration_count = 0;

while true
    iteration_count = iteration_count + 1;

    R_bar = R - q;
    squared_sums = sum(R_bar.^2, 1);
    column_norms = sqrt(squared_sums);

    weights = w_test(column_norms,k,h);

    R_bar = R_bar .* sqrt(weights);

    X = R_bar' * U;

    X_bar = [ones(N, 1), X];

    alpha=SbySVD(X_bar' * X_bar,X_bar' * R_bar');

    q_bar = q + alpha(1, :)';

   [Q, ~, ~] = qr(alpha(2:end, :)', 0);

    U = Q(:,1:d);
    z=U' * (r - q_bar);
    q = q_bar + U * z;

    if norm(q - q_prev) < epsilon || iteration_count >= max_iterations
        break;
    end

    q_prev = q;
end
