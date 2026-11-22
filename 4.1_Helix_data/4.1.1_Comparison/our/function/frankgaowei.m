function f = frankgaowei(X)
% Evaluate the modified Franke function at column samples.
    if isvector(X)
        X = reshape(X, 1, []);
    end
    x = X(1, :);  
    y = X(2, :);  
    z = X(3, :);  
    f = 3/4 * exp(-((9*x - 2).^2 + (9*y - 2).^2 + (z-2))/4) ...
        + 3/4 * exp(-((9*x - 1).^2)/49 - ((9*y + 1).^2)/10 + ((z).^2)/10) ...
        - 1/5 * exp(-((9*x - 4).^2 + (9*y - 7).^2 + (z).^2)) ...
        + 1/2 * exp(-((9*x - 7).^2 + (9*y - 3).^2 + (z-2))/4);
    f = f(:)';
end
