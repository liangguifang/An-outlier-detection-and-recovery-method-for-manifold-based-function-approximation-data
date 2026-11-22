function qTilde0 = MMLSProjectedSeparation(geometry,outlierIndex)
% Half the minimum outlier-pair distance over all sample local spaces.
% Use either cached coordinates x_k^(i) or the samples and local bases.

hasCoordinates = isfield(geometry,'projectedCoordinates');
if hasCoordinates
    N = size(geometry.projectedCoordinates,2);
elseif isfield(geometry,'localBases') && isfield(geometry,'samples')
    N = size(geometry.samples,2);
else
    error('MMLSProjectedSeparation:MissingCoordinates', ...
        'Geometry must contain projected coordinates or samples and local bases.');
end
if isempty(outlierIndex)
    qTilde0 = NaN;
    return;
end
validateattributes(outlierIndex,{'numeric'}, ...
    {'vector','integer','positive','<=',N},mfilename,'outlierIndex');
outlierIndex = unique(outlierIndex(:));
K = numel(outlierIndex);
if K < 2
    qTilde0 = NaN;
    return;
end

if hasCoordinates
    coordinates = geometry.projectedCoordinates(:,outlierIndex,:);
else
    points = geometry.samples(:,outlierIndex);
    % The origin cancels in pair differences. Center at one outlier before projection.
    points = bsxfun(@minus,points,points(:,1));
    coordinates = pagemtimes(geometry.localBases,'transpose',points,'none');
end

d = size(coordinates,1);
numSpaces = size(coordinates,3);
if d == 1
    % In one dimension the closest pair is adjacent after sorting.
    gaps = diff(sort(reshape(coordinates,K,numSpaces),1),1,1);
    qTilde0 = 0.5*min(gaps(:));
    return;
end

% The first-coordinate gap bounds the full Euclidean distance from below.
[sortedFirst, order] = sort(reshape(coordinates(1,:,:),K,numSpaces),1);
orderedColumns = bsxfun(@plus,order,K*(0:numSpaces-1));
points = reshape(coordinates,d,[]);
left = orderedColumns(1:end-1,:);
right = orderedColumns(2:end,:);
differences = points(:,right(:))-points(:,left(:));
minSquaredDistance = min(sum(differences.^2,1));

for lag = 2:K-1
    gaps = sortedFirst(1+lag:end,:)-sortedFirst(1:end-lag,:);
    [firstRank, space] = find(gaps.^2 < minSquaredDistance);
    if isempty(firstRank)
        break;  % Larger lags only increase the gap at each starting rank.
    end
    left = firstRank + K*(space-1);
    differences = points(:,orderedColumns(left+lag))-points(:,orderedColumns(left));
    minSquaredDistance = min(minSquaredDistance,min(sum(differences.^2,1)));
end
qTilde0 = 0.5*sqrt(minSquaredDistance);
end
