function [A,neighborCount,localRadii,geometry,Btest] = buildLocalPcaMLS(X,d,m,sparseFactor,queryCentered,cacheProjectedCoordinates,testPoints,searchRadius)
    % 按 manapprox 的原空间半径查询选点。neighborCount 记录每一行实际使用的点数。
    % 第四个输出包含训练几何。第六个参数为 false 时只缓存局部基底。
    % testPoints 每行一个查询点。Btest 只使用 X 中的训练样本构造。
    % 可指定 searchRadius，以便在敏感性分析中独立改变搜索半径。
    if nargin < 4 || isempty(sparseFactor)
        sparseFactor = 3;
    end
    % 查询点邻域模式在 FLCS 和多项式拟合中使用同一组样本。
    if nargin < 5
        queryCentered = false;
    end
    if nargin < 6
        cacheProjectedCoordinates = true;
    end
    queries = X;
    if nargin >= 7 && ~isempty(testPoints)
        validateattributes(testPoints,{'numeric'},{'2d','real','finite','ncols',size(X,2)}, ...
            mfilename,'testPoints');
        if ~queryCentered
            error('buildLocalPcaMLS:QueryCenteredRequired', ...
                'Test points require query-centered fitting.');
        end
        queries = [X;testPoints];
    end
    validateattributes(sparseFactor,{'numeric'},{'scalar','real','finite','positive'}, ...
        mfilename,'sparseFactor');

    [Alpha,Q] = PolyAlpha(d,m);
    N = size(X,1);
    numQueries = size(queries,1);
    maxFitNeighbors = 30;
    if Q > maxFitNeighbors
        error('buildLocalPcaMLS:TooManyBasisTerms', ...
            'Degree %d in dimension %d needs %d basis terms, exceeding the %d-point fitting limit.', ...
            m,d,Q,maxFitNeighbors);
    end
    if N < Q
        error('buildLocalPcaMLS:NotEnoughSamples', ...
            'Degree %d requires at least %d training samples.',m,Q);
    end

    if nargin < 8 || isempty(searchRadius)
        sigma = estimateSearchRadius(X,sparseFactor,Q);
    else
        validateattributes(searchRadius,{'numeric'},{'scalar','real','finite','positive'}, ...
            mfilename,'searchRadius');
        sigma = searchRadius;
    end
    radius = 1.5*sigma;  % manapprox 中紧支撑权函数的半径
    columns = cell(numQueries,1);
    weights = cell(numQueries,1);
    neighborCount = zeros(numQueries,1);
    flcsConverged = false(numQueries,1);
    localRadii = repmat(radius,numQueries,1);
    storeGeometry = nargout >= 4;
    if storeGeometry
        projectedDistances = zeros(N,N);
        if cacheProjectedCoordinates
            projectedCoordinates = zeros(d,N,N);
        else
            localBases = zeros(size(X,2),d,N);
        end
    end
    p0 = zeros(Q,1);
    p0(1) = 1;  % PolyAlpha 的第一项为常数，查询点投影到原点。
    projectionTolerance = 1e-6;
    weightTolerance = 1e-13;  % ManApprox.EPSILON
    R = X.';
    for j = 1:numQueries
        query = queries(j,:).';
        rowRadius = radius;
        if queryCentered
            queryOffset = bsxfun(@minus,X,query.');
            queryDistanceSquared = sum(queryOffset.^2,2);
            sortDistanceSquared = queryDistanceSquared;
            if j <= N
                sortDistanceSquared(j) = -1;  % 重合点也优先保留训练查询点自身。
            end
            ids = selectFitNeighbors(sortDistanceSquared,sigma,maxFitNeighbors);
            if numel(ids) < Q
                if j <= N
                    error('buildLocalPcaMLS:InsufficientQueryNeighbors', ...
                        'Training point %d has %d neighbors at sigma=%.6g but needs %d. Increase sparseFactor.', ...
                        j,numel(ids),sigma,Q);
                end
                % 测试点附近的半径内样本不足时，按原空间距离补足训练近邻。
                [~,order] = sort(queryDistanceSquared,'ascend');
                ids = order(1:min(N,maxFitNeighbors));
            end
            rowRadius = 1.5*sqrt(max(queryDistanceSquared(ids)));
            if rowRadius <= 0
                error('buildLocalPcaMLS:ZeroLocalRadius', ...
                    'All selected neighbors coincide with query point %d.',j);
            end
            % FLCS 内部权重半径为 4*h。只传所选样本，使邻域固定。
            [basis,origin,flcsConverged(j)] = FLCS(R(:,ids),query,projectionTolerance,d,rowRadius/4);
        else
            [basis,origin,flcsConverged(j)] = FLCS(R,query,projectionTolerance,d,radius/4,sigma);
        end
        localRadii(j) = rowRadius;
        offset = bsxfun(@minus,X,origin.');
        if ~queryCentered
            ids = selectFitNeighbors(sum(offset.^2,2),sigma,maxFitNeighbors);
        end
        xy = offset(ids,:)*basis/rowRadius;
        if storeGeometry && j <= N
            projection = basis.'*offset.';
            if cacheProjectedCoordinates
                projectedCoordinates(:,:,j) = projection;
            else
                localBases(:,:,j) = basis;
            end
            relativeProjection = bsxfun(@minus,projection,projection(:,j));
            projectedDistances(j,:) = sqrt(sum(relativeProjection.^2,1));
        end
        w = compactWeight(sqrt(sum(xy.^2,2)));
        active = w > weightTolerance;
        ids = ids(active);
        xy = xy(active,:);
        sqrtW = sqrt(w(active));
        k = numel(ids);
        neighborCount(j) = k;
        if k < Q
            error('buildLocalPcaMLS:InsufficientNeighbors', ...
                'Query point %d has %d active neighbors at sigma=%.6g but needs %d. Increase sparseFactor.', ...
                j,k,sigma,Q);
        end

        P = ones(k,Q);
        for ell = 1:d
            P = P.*bsxfun(@power,xy(:,ell),Alpha(:,ell).');
        end
        design = bsxfun(@times,P,sqrtW);
        gram = design.'*design;
        if rank(gram) < Q
            error('buildLocalPcaMLS:RankDeficientNormalMatrix', ...
                'Query point %d has a numerically singular degree-%d normal matrix.',j,m);
        end

        beta = SbySVD(gram,p0);
        weights{j} = sqrtW.*(design*beta);
        columns{j} = ids;
    end
    rows = repelem((1:numQueries).',neighborCount);
    allWeights = sparse(rows(:),vertcat(columns{:}),vertcat(weights{:}),numQueries,N);
    A = allWeights(1:N,:);
    if nargout >= 5
        Btest = allWeights(N+1:end,:);
    end
    neighborCount = neighborCount(1:N);
    localRadii = localRadii(1:N);
    if any(~flcsConverged)
        warning('buildLocalPcaMLS:FLCSIterationLimit', ...
            'FLCS reached the iteration limit without meeting the tolerance at %d of %d query points.', ...
            nnz(~flcsConverged),numQueries);
    end
    if storeGeometry
        reverseDistances = projectedDistances.';
        withinRadius = bsxfun(@lt,projectedDistances,localRadii);
        mask = withinRadius | withinRadius.';
        geometry.sigma = sigma;
        geometry.flcsConverged = flcsConverged(1:N);
        if cacheProjectedCoordinates
            geometry.projectedCoordinates = projectedCoordinates;
        else
            geometry.localBases = localBases;
            geometry.samples = R;
        end
        geometry.omega_delta = max(abs(projectedDistances(mask)-reverseDistances(mask)));
    end
end

function ids = selectFitNeighbors(distanceSquared,sigma,maxFitNeighbors)
    ids = find(distanceSquared < sigma^2);
    if numel(ids) > maxFitNeighbors
        [~,order] = sort(distanceSquared(ids),'ascend');
        ids = ids(order(1:maxFitNeighbors));
    end
end

function sigma = estimateSearchRadius(X,sparseFactor,Q)
    % ManApprox.calculateSigma：100 个参考点的原空间距离分位数取最大值。
    N = size(X,1);
    p = min((sparseFactor*Q+1)/N,1);
    stream = RandStream('mt19937ar','Seed',1);
    pool = 4:N-3;  % Python 的 randint(3,N-3) 对应的 MATLAB 下标
    if isempty(pool)
        pool = 1:N;
    end
    anchors = pool(randi(stream,numel(pool),100,1));
    % NumPy percentile 的默认线性插值位置。
    position = 1+(N-1)*p;
    lo = floor(position);
    hi = ceil(position);
    sigma = 0;
    for j = 1:numel(anchors)
        offset = bsxfun(@minus,X,X(anchors(j),:));
        sorted = sort(sqrt(sum(offset.^2,2)));
        sampleRadius = sorted(lo)+(position-lo)*(sorted(hi)-sorted(lo));
        sigma = max(sigma,sampleRadius);
    end
    if ~isfinite(sigma) || sigma <= 0
        error('buildLocalPcaMLS:InvalidSearchRadius','Estimated neighborhood radius must be positive and finite.');
    end
end

function w = compactWeight(t)
    w = zeros(size(t));
    inside = t < 1;
    w(inside) = exp(-(t(inside)./(1-t(inside))).^2);
end
