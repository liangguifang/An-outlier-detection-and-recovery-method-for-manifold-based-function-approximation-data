% Race car OMP residual curves for the two stopping thresholds.
clear; clc; close all;
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir,'function'));
% 使用 car100 保存的矩阵和成对的图像角度。
matFile = fullfile(scriptDir,'car100_A_N60.mat');
S = load(matFile,'A','f_true','eligibleIdx','selectedFrameNumbers','localRadii');
requiredFields = {'A','f_true','eligibleIdx','selectedFrameNumbers','localRadii'};
if ~all(isfield(S,requiredFields))
    error('The saved car100 matrix is missing frame or neighborhood data. Rebuild A with car100.');
end
A = sparse(S.A);
N = size(A,1);
f = S.f_true(:);
eligibleIdx = S.eligibleIdx(:);
if ~isequal(size(A),[60 60]) || ~isequal(S.selectedFrameNumbers(:),(6:65)') || numel(f)~=N
    error('carEpstu requires the corrected car100 matrix for frames 6 through 65.');
end
if numel(S.localRadii)~=N || min(S.localRadii)==max(S.localRadii)
    error('The saved matrix does not match the current query-centered neighborhood settings.');
end

U = speye(N)-A;
UN = full(sqrt(sum(U.^2,1))');
Um = min(UN);
if any(UN <= 0) || any(~isfinite(UN))
    error('U contains a zero or nonfinite column norm.');
end
Unormalized = bsxfun(@rdivide,full(U),UN');
coherenceMatrix = abs(Unormalized'*Unormalized);
coherenceMatrix(1:N+1:end) = 0;
fprintf('Maximum column coherence of U = %.6e\n',max(coherenceMatrix(:)));

KList = [4 8 12];
curveColors = [0 0.4470 0.7410; 0.9290 0.6940 0.1250; 0.4940 0.1840 0.5560];
thresholds = [0.05 0.50];
trialsPerGroup = 100;
outlierMagnitude = [5,10];
outlierStream = RandStream('mt19937ar','Seed',1);
RnList = cell(size(KList));
stopKList = nan(numel(KList),numel(thresholds));
for g = 1:numel(KList)
    K = KList(g);
    if K > numel(eligibleIdx)
        error('K cannot exceed the number of eligible images.');
    end
    % 画出 car100 中每个 K 的第一轮，保持三个 K 共用的随机序列。
    [~,eligibleOutliers] = MMLSSignedOutliers(outlierStream,numel(eligibleIdx),K,outlierMagnitude);
    E = zeros(N,1);
    E(eligibleIdx) = eligibleOutliers;
    for trial = 2:trialsPerGroup
        MMLSSignedOutliers(outlierStream,numel(eligibleIdx),K,outlierMagnitude);
    end
    f_observed = f+E;
    eta = full(Um*abs(median(f_observed)));
    for j = 1:numel(thresholds)
        [~,~,stoppedResiduals] = OMPBYzspForL0(U,U*f_observed,N,eta,thresholds(j));
        stopKList(g,j) = numel(stoppedResiduals)-1;
    end
    % Disable early stopping to record the available residual trajectory.
    [~,~,Rn] = OMPBYzspForL0(U,U*f_observed,61,0,-Inf);
    Rn = full(Rn(:));
    if any(~isfinite(Rn)) || any(Rn < 0)
        error('OMP returned invalid residual norms.');
    end
    RnList{g} = Rn;
    fprintf('K=%d, ||U*f||=%.6e, ||U*E||=%.6e, ||U*f_obs||=%.6e\n', ...
        K,norm(U*f),norm(U*E),norm(U*f_observed));
end

% All three curves use the SAME residual-to-display transformation.
splitK = 58;
lowerHeight = 0.10;
allResiduals = vertcat(RnList{:});
logMin = log10(max(min(allResiduals),realmin));
logMax = log10(max(allResiduals));
splitResiduals = zeros(size(KList));
for g = 1:numel(KList)
    if numel(RnList{g}) <= splitK
        error('Each trajectory must reach k=58 for this display scale.');
    end
    splitResiduals(g) = RnList{g}(splitK+1);
end
% The largest k=58 residual defines the shared tail boundary.
logSplit = log10(max(max(splitResiduals),realmin));
if logSplit <= logMin || logSplit >= logMax
    error('The shared tail boundary must lie inside the residual range.');
end
scaleKnots = [logMin logSplit logMax];
displayKnots = [0 lowerHeight 1];
if logSplit < 0 && logMax > log10(11)
    scaleKnots = [logMin logSplit 0 log10(5) log10(11) logMax];
    displayKnots = [0 lowerHeight lowerHeight+0.06 ...
        lowerHeight+0.12 lowerHeight+0.18 1];
end
tickValues = unique([1e-14 1e-10 1e-6 1e-2 1 5 11:2:floor(max(allResiduals))]);
tickLogs = log10(tickValues);
tickLogs = tickLogs(tickLogs >= logMin & tickLogs <= logMax);
tickPositions = interp1(scaleKnots,displayKnots,tickLogs);
keepTicks = false(size(tickPositions));
lastTickPosition = -Inf;
for j = 1:numel(tickPositions)
    if tickPositions(j)-lastTickPosition >= 0.04
        keepTicks(j) = true;
        lastTickPosition = tickPositions(j);
    end
end
tickLogs = tickLogs(keepTicks);
tickPositions = tickPositions(keepTicks);

fig = figure('Color','w','Position',[120 120 1050 680]);
ax = axes('Parent',fig);
hold(ax,'on');
for g = 1:numel(KList)
    Rn = RnList{g};
    k_record = (0:numel(Rn)-1)';
    displayResidual = interp1(scaleKnots,displayKnots,log10(max(Rn,realmin)));
    plot(ax,k_record,displayResidual,'-o','Color',curveColors(g,:), ...
        'LineWidth',1.4,'MarkerSize',3, ...
        'DisplayName',sprintf('K = %d',KList(g)));
end
% Hollow circles denote 5%, and hollow triangles denote 50%.
% Start with size 13, enlarging only markers whose screen positions overlap.
baseMarkerSize = 13;
stopMarkerSizes = baseMarkerSize*ones(size(stopKList));
stopY = nan(size(stopKList));
for j = 1:numel(thresholds)
    for g = 1:numel(KList)
        if isfinite(stopKList(g,j))
            Rn = RnList{g};
            stopY(g,j) = interp1(scaleKnots,displayKnots, ...
                log10(max(Rn(stopKList(g,j)+1),realmin)));
        end
    end
end
xlim(ax,[0 61]); ylim(ax,[-0.02 1.02]);
axesPixels = getpixelposition(ax);
markerDiameterPixels = baseMarkerSize*get(groot,'ScreenPixelsPerInch')/72;
validMarkers = find(isfinite(stopKList) & isfinite(stopY));
for q = 2:numel(validMarkers)
    current = validMarkers(q);
    previous = validMarkers(1:q-1);
    dx = (stopKList(previous)-stopKList(current))*axesPixels(3)/61;
    dy = (stopY(previous)-stopY(current))*axesPixels(4)/1.04;
    overlaps = previous(hypot(dx,dy) < markerDiameterPixels);
    if ~isempty(overlaps)
        stopMarkerSizes(current) = max(stopMarkerSizes(overlaps))+4;
    end
end
stopMarkers = {'o','^'};
for j = 1:numel(thresholds)
    % Draw larger markers first so smaller markers remain visible.
    for g = numel(KList):-1:1
        kStop = stopKList(g,j);
        if isfinite(kStop)
            Rn = RnList{g};
            yStop = interp1(scaleKnots,displayKnots,log10(max(Rn(kStop+1),realmin)));
            markerFace = 'none';
            plot(ax,kStop,yStop,stopMarkers{j},'LineStyle','none', ...
                'MarkerSize',stopMarkerSizes(g,j),'MarkerFaceColor',markerFace, ...
                'MarkerEdgeColor',curveColors(g,:),'LineWidth',1.6, ...
                'Clipping','off', ...
                'HandleVisibility','off');
        end
    end
    legendFace = 'none';
    plot(ax,NaN,NaN,stopMarkers{j},'LineStyle','none', ...
        'MarkerSize',baseMarkerSize,'MarkerFaceColor',legendFace, ...
        'MarkerEdgeColor','k','LineWidth',1.6, ...
        'DisplayName',sprintf('\\tau = %.0f%%',100*thresholds(j)));
end
grid(ax,'on'); box(ax,'on');
xlim(ax,[0 61]); ylim(ax,[-0.02 1.02]);
halfThresholdIndex = find(thresholds==0.50,1);
halfStops = stopKList(:,halfThresholdIndex);
halfStops = halfStops(isfinite(halfStops));
xTicks = unique([0:10:60 4 7 halfStops']);
xticks(ax,xTicks);
xticklabels(ax,compose('%d',xTicks));
yticks(ax,tickPositions);
yticklabels(ax,compose('%.2g',10.^tickLogs));
xlabel(ax,'Iteration k');
ylabel(ax,'$\left\|\mathbf{v}^{(i)}\right\|_2$', ...
    'Interpreter','latex','FontSize',14);
title(ax,'Rotating race car \tau analysis','Interpreter','tex');
legend(ax,'Location','northeast','Interpreter','tex','FontSize',9);
ax.XMinorGrid = 'off';
ax.YMinorGrid = 'off';
print(fig,fullfile(scriptDir,'car_nr_vs_k.png'),'-dpng','-r300');
if exist('exportgraphics','file') == 2
    exportgraphics(fig,fullfile(scriptDir,'car_nr_vs_k.pdf'),'ContentType','vector');
end
if isgraphics(fig,'figure')
    savefig(fig,fullfile(scriptDir,'car_nr_vs_k.fig'));
end
