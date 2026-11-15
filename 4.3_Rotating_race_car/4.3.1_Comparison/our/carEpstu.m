% carEpstu.m
% Compare OMP residual trajectories for K=4, 8, and 12.
clear; clc; close all;
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir,'function'));
matFile = fullfile(scriptDir,'carresults_k=6.mat');
S = load(matFile,'A');
if ~isfield(S,'A')
    error('The MAT file must contain the matrix A.');
end
A = sparse(S.A);
N = size(A,1);

imageFolder = fullfile(scriptDir,'car');
imageFiles = dir(fullfile(imageFolder,'*.png'));
totalImages = numel(imageFiles);
if totalImages < N
    error('The car image folder contains fewer images than A requires.');
end
allAngles = (0:totalImages-1)'*(2*pi/totalImages);
startIdx = floor((totalImages-N)/2)+1;
selectedIdx = startIdx:(startIdx+N-1);
f = allAngles(selectedIdx);

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
thresholds = [0.05 0.10 0.20 0.22 0.24 0.26 0.28 0.30 0.50 0.55];
Eps = 0.5;
RnList = cell(size(KList));
stopKList = nan(numel(KList),numel(thresholds));
for g = 1:numel(KList)
    K = KList(g);
    if K > N
        error('K cannot exceed the number of images.');
    end
    rng(1);
    index = sort(randperm(N,K));
    E = zeros(N,1);
    for j = 1:K
        magnitude = 5 + 5*rand(1);
        direction = 2*randi([0,1])-1;
        E(index(j)) = direction*magnitude;
    end
    f_observed = f+E;
    eta = full(Um*abs(median(f_observed)));
    [Ep,Index] = OMPBYzspForL0(U,U*f_observed,N,eta,Eps);
    % Disable early stopping to record the available residual trajectory.
    [~,~,Rn] = OMPBYzspForL0(U,U*f_observed,61,0,-Inf);
    Rn = full(Rn(:));
    if any(~isfinite(Rn)) || any(Rn < 0)
        error('OMP returned invalid residual norms.');
    end
    RnList{g} = Rn;
    fprintf('K=%d, ||U*f||=%.6e, ||U*E||=%.6e, ||U*f_obs||=%.6e\n', ...
        K,norm(U*f),norm(U*E),norm(U*f_observed));
    ratio = abs(Rn(2:end-1)-Rn(3:end))./ ...
        abs(Rn(1:end-2)-Rn(2:end-1));
    for j = 1:numel(thresholds)
        idx = find(ratio <= thresholds(j),1,'first');
        if ~isempty(idx)
            stopKList(g,j) = idx+1;
        end
    end
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
% Thresholds with the same stopping pattern across all K share one label.
stopPatterns = stopKList';
stopPatterns(~isfinite(stopPatterns)) = -1;
[~,~,criterionGroup] = unique(stopPatterns,'rows');
criterionColors = lines(max(criterionGroup));
for j = 1:max(criterionGroup)
    useThresholds = criterionGroup == j;
    tauText = strjoin(compose('%.2g',thresholds(useThresholds)),', ');
    firstThreshold = find(useThresholds,1);
    for g = 1:numel(KList)
        kStop = stopKList(g,firstThreshold);
        if isfinite(kStop)
            Rn = RnList{g};
            yStop = interp1(scaleKnots,displayKnots,log10(max(Rn(kStop+1),realmin)));
            plot(ax,kStop,yStop,'p','LineStyle','none', ...
                'MarkerSize',15,'MarkerFaceColor',criterionColors(j,:), ...
                'MarkerEdgeColor',criterionColors(j,:), ...
                'HandleVisibility','off');
        end
    end
    plot(ax,NaN,NaN,'p','LineStyle','none', ...
        'MarkerSize',15,'MarkerFaceColor',criterionColors(j,:), ...
        'MarkerEdgeColor',criterionColors(j,:), ...
        'DisplayName',sprintf('\\tau = [%s]',tauText));
end
grid(ax,'on'); box(ax,'on');
xlim(ax,[0 61]); ylim(ax,[-0.02 1.02]);
xticks(ax,0:10:60);
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

