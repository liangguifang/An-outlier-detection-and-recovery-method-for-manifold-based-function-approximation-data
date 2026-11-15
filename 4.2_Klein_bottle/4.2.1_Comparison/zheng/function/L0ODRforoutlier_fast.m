function [Ep,Index,f_approx] = L0ODRforoutlier_fast(P,f,Eps)
% Faster OMP using incremental QR instead of refactoring At'*At each step.
% P = L0ODRprepare(A), called ONCE before the trials.
% Keeps correlation selection, eta, Eps and Max/4 iteration limit.
% Preserves original behavior of discarding the last selected atom
% if the residual-ratio criterion triggers a break.
% QR and normal equations may produce different rounding near singularity.

U = P.U;
n = size(U,1);
f = f(:);
if numel(f) ~= n || ~isreal(f) || any(~isfinite(f))
    error('f must be a finite real n-by-1 vector.');
end
if nargin < 3 || isempty(Eps) || ~isscalar(Eps) || ...
        ~isfinite(Eps) || Eps < 0
    error('Eps must be a finite nonnegative scalar.');
end

f_approx = P.A * f;
R = U * f;
r = R;
nr = norm(r);
Ep = zeros(n,1);
Index = [];
if nr <= P.eta
    return;
end

maxSteps = ceil(n/4);
Q = zeros(n,maxSteps);
T = zeros(maxSteps,maxSteps);
proj = zeros(maxSteps,1);
selected = zeros(maxSteps,1);
available = true(n,1);
valid = (P.normU > 0);
Rn = zeros(maxSteps+1,1);
Rn(1) = nr;
kKeep = 0;

for k = 1:maxSteps
    if nr <= P.eta
        break;
    end

    scores = abs(U.' * r) ./ P.normU;
    scores(~available | ~valid) = -Inf;
    [best,id] = max(scores);

    if ~isfinite(best)
        break;
    end

    a = U(:,id);
    v = a;

    if k > 1
        H = Q(:,1:k-1).' * v;
        v = v - Q(:,1:k-1) * H;

        H2 = Q(:,1:k-1).' * v;
        v = v - Q(:,1:k-1) * H2;
        H = H + H2;
        T(1:k-1,k) = H;
    end

    diagT = norm(v);
    if diagT <= 1e-12*norm(a)
        error('L0ODRfast:RankDeficient', ...
            'Selected OMP columns are nearly dependent (step %d).',k);
    end

    T(k,k) = diagT;
    Q(:,k) = v/diagT;
    proj(k) = Q(:,k).' * R;
    selected(k) = id;
    available(id) = false;

    r = r - Q(:,k) * (Q(:,k).' * r);
    nr = norm(r);
    Rn(k+1) = nr;

    if k > 1
        numerator = abs(Rn(k)-Rn(k+1));
        denominator = abs(Rn(k-1)-Rn(k));

        if denominator == 0
            if numerator == 0
                ratio = 0;
            else
                ratio = Inf;
            end
        else
            ratio = numerator/denominator;
        end

        if ratio <= Eps
            break;
        end
    end

    kKeep = k;
end

if kKeep > 0
    coeff = T(1:kKeep,1:kKeep) \ proj(1:kKeep);
    Ep(selected(1:kKeep)) = coeff;
    Index = sort(selected(1:kKeep));
end
end