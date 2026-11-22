% Rotating race car detection and recovery benchmark.
clearvars;
clc;
overallClock = tic;

%% 0. 参数和函数路径
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
addpath(scriptDir);

functionDir = fullfile(scriptDir, 'function');
if isfolder(functionDir)
    addpath(functionDir);
end

imageFolder = fullfile(scriptDir,'..','..','..','data','car');

% false：原 car.m 的设置，直接使用中间 60 张图；
% true ：使用全部图（通常为 72 张），只在中间 60 张加入异常值。

useAllImages = false;
numCenterImages = 60;

epsilon = 1e-6;
d = 1;
m = 3;
hManual = [];  % []：以最大最近邻距离估计 h；也可指定已有的 h。

KList = [4, 8, 12];
T = 100;
ompTau = 0.5;  % OMP 相邻残差降幅比的停止阈值。

if ~isscalar(T) || T < 1 || T ~= floor(T)
    error('T 必须是正整数。');
end
if isempty(KList) || any(KList < 1) || any(KList ~= floor(KList))
    error('KList 必须包含正整数。');
end

%% 1. 加载图像和真实角度（只读取一次）
if ~isfolder(imageFolder)
    error('图像文件夹不存在：%s', imageFolder);
end
imageFiles = dir(fullfile(imageFolder, '*.png'));
totalImages = numel(imageFiles);

if totalImages < 2
    error('至少需要 2 张 PNG 图像。');
end

% 文件名按字符串排序时，帧 10 会排在帧 2 前。按末尾数字恢复旋转顺序。
frameTokens = regexpi({imageFiles.name},'(\d+)\.png$','tokens','once');
if any(cellfun('isempty',frameTokens))
    error('每张图像的文件名必须以数字帧号和 .png 结尾。');
end
frameNumbers = cellfun(@(token) str2double(token{1}),frameTokens);
[frameNumbers,frameOrder] = sort(frameNumbers(:));
if any(diff(frameNumbers) ~= 1)
    error('图像帧号必须唯一且连续，才能按等角度间隔赋值。');
end
imageFiles = imageFiles(frameOrder);
allAngles = (frameNumbers-frameNumbers(1)) * (2*pi/totalImages);

numCenterImages = min(numCenterImages, totalImages);
startIdx = floor((totalImages - numCenterImages)/2) + 1;
centerGlobalIdx = startIdx:(startIdx + numCenterImages - 1);

if useAllImages
    selectedIdx = (1:totalImages)';
    eligibleIdx = centerGlobalIdx(:);
else
    selectedIdx = centerGlobalIdx(:);
    eligibleIdx = (1:numel(selectedIdx))';
end
selectedFiles = imageFiles(selectedIdx);
selectedFrameNumbers = frameNumbers(selectedIdx);
f_true = allAngles(selectedIdx);

N = numel(selectedFiles);
if any(KList > numel(eligibleIdx))
    error('异常值数量不能超过可选图像数量 %d。', numel(eligibleIdx));
end

firstImage = im2double(imread(fullfile(imageFolder, selectedFiles(1).name)));
if ~ismatrix(firstImage)
    error('图片必须为灰度图像。');
end

r = zeros(numel(firstImage), N);
r(:,1) = firstImage(:);

for i = 2:N
    img = im2double(imread(fullfile(imageFolder, selectedFiles(i).name)));
    if ~ismatrix(img) || ~isequal(size(img), size(firstImage))
        error('图像 %s 不是同尺寸灰度图。', selectedFiles(i).name);
    end
    r(:,i) = img(:);
end

fprintf('N = %d, 图像维数 = %d, 可放置异常值的样本数 = %d\n', ...
    N, size(r,1), numel(eligibleIdx));
fprintf('物理帧号 = %d 到 %d，角度 = %.1f 到 %.1f 度\n', ...
    selectedFrameNumbers(1),selectedFrameNumbers(end),rad2deg(f_true(1)),rad2deg(f_true(end)));

%% 2. 计算图像距离矩阵、h、支撑半径
% 两张图的欧氏距离与 KB100.m 中使用的点间距定义一致。
% 预先计算距离矩阵，避免在 100 次循环中重新做高维运算。
squareNorms = sum(r.^2, 1);
D2 = bsxfun(@plus, squareNorms(:), squareNorms(:)') - 2*(r'*r);
D2 = max(D2, 0);

pairwiseDist = sqrt(D2);

if isempty(hManual)
    D2(1:N+1:end) = inf;  % 计算最近邻时排除自己
    h = sqrt(max(min(D2, [], 2)));
else
    h = hManual;
end

if ~isscalar(h) || ~isreal(h) || ~isfinite(h) || h <= 0
    error('h 必须是正的有限实数。');
end

fprintf('Reference h = %.6e\n',h);

%% 3. 构造 A：只执行一次
fprintf('开始构造 MMLS 矩阵 A ...\n');
drawnow;

buildClock = tic;
% 直接使用原始图像向量，以查询图像为中心固定邻域。
[A, neighborCount, localRadii, geometry] = buildLocalPcaMLS(r.',d,m,3,true);
% 用最大局部半径作为分离条件中的统一半径上界。
support_radius = max(localRadii);
delta = support_radius;
omega_delta = geometry.omega_delta;
tA = toc(buildClock);

if ~isequal(size(A), [N, N]) || any(~isfinite(A(:)))
    error('buildLocalPcaMLS 返回的 A 尺寸不正确或存在 NaN/Inf。');
end

fprintf('A 构造完成：tA = %.6e s\n', tA);
fprintf('Neighbors per row = %d to %d, search radius sigma = %.6e\n', ...
    min(neighborCount),max(neighborCount),geometry.sigma);

% 构造完成即单独保存 A，无需等待后续随机异常实验结束。
AFile = fullfile(scriptDir, sprintf('car100_A_N%d.mat',N));
save(AFile, 'A', 'geometry', 'N', 'f_true', 'h', 'delta', ...
    'support_radius', 'neighborCount', 'localRadii', 'epsilon', 'd', 'm', 'tA', ...
    'selectedFiles', 'selectedFrameNumbers', 'eligibleIdx', 'centerGlobalIdx', ...
    'useAllImages', 'numCenterImages', 'imageFolder', '-v7.3');
fprintf('A 已单独保存至：%s\n',AFile);

% 后面只需要小型的图像间距矩阵，不再需要原始高维图像。
clear r D2 squareNorms firstImage img;

%% 4. 随机异常实验：每个 K 做 T 次
% 列依次为 TP, FP, FN, Recall, Precision, F1, Error, SolveSeconds
metricNames = {'TP','FP','FN','Recall','Precision','F1', ...
               'Error','SolveSeconds'};

results = struct([]);
resultFile = fullfile(scriptDir, 'car_results_split_k=6.mat');

% 放在 A 构造之后，避免 svds 等操作影响异常值的随机序列。
rng(1,'twister');

trialClock = tic;

for g = 1:numel(KList)
    K = KList(g);

    TP_all = zeros(T,1);
    FP_all = zeros(T,1);
    FN_all = zeros(T,1);

    Recall_all = zeros(T,1);
    Precision_all = zeros(T,1);
    F1_all = zeros(T,1);
    Error_all = zeros(T,1);

    FullDetection_all = false(T,1);
    Q0_all = nan(T,1);
    outlierIndices = zeros(K,T);
    SolveSeconds_all = zeros(T,1);

    for trial = 1:T
        % 每次试验都从干净的角度值重新开始。
        E = zeros(N,1);
        index = sort(eligibleIdx(randperm(numel(eligibleIdx), K)));
        outlierIndices(:,trial) = index(:);

        % 与 Zhao 使用相同随机调用顺序：逐点先生成幅值，再生成符号。
        for j = 1:K
            magnitude = 5 + 5*rand(1);
            direction = 2*randi([0,1])-1;
            E(index(j)) = direction*magnitude;
        end
        f = f_true + E;

        % 只重复 OMP 与函数拟合，不重新构造 A。
        solveClock = tic;
        [Ep, Index, f_approx] = MMLSforoutlier(A, f, [], [], ompTau);
        SolveSeconds_all(trial) = toc(solveClock);

        ep_indices = find(Ep(:) ~= 0);

        TP = numel(intersect(index, ep_indices));
        FP = numel(setdiff(ep_indices,index));
        FN = numel(setdiff(index,ep_indices));
        FullDetection_all(trial) = (TP == K && FP == 0 && FN == 0);

        TP_all(trial) = TP;
        FP_all(trial) = FP;
        FN_all(trial) = FN;

        if TP + FN == 0
            Recall = 0;
        else
            Recall = TP / (TP + FN);
        end

        if TP + FP == 0
            Precision = 0;
        else
            Precision = TP / (TP + FP);
        end

        if 2*TP + FP + FN == 0
            F1 = 0;
        else
            F1 = 2*TP / (2*TP + FP + FN);
        end

        Recall_all(trial) = Recall;
        Precision_all(trial) = Precision;
        F1_all(trial) = F1;

        Error_all(trial) = norm(Ep(:) - E, 2) / K;

        if mod(trial, 10) == 0 || trial == 1
            fprintf('K=%d：第 %d/%d 次完成，总用时 %.1f s\n', ...
                K, trial, T, toc(trialClock));
        end
    end

    %% 5. 当前 K 的均值与标准差（对应 KB100.m）
    Recall_mean = mean(Recall_all);
    Recall_std = std(Recall_all);

    Precision_mean = mean(Precision_all);
    Precision_std = std(Precision_all);

    F1_mean = mean(F1_all);
    F1_std = std(F1_all);

    Error_mean = mean(Error_all);
    Error_std = std(Error_all);

    FullDetection_count = sum(FullDetection_all);
    % 使用当前 K 的各轮真实异常位置，在识别结束后计算分离距离。
    for trial = 1:T
        Q0_all(trial) = MMLSProjectedSeparation(geometry,outlierIndices(:,trial));
    end
    Q0Condition_all = support_radius + omega_delta <= Q0_all;
    Q0Ratio_all = Q0_all/(support_radius+omega_delta);
    Q0Ratio_mean = mean(Q0Ratio_all,'omitnan');
    Q0Ratio_std = std(Q0Ratio_all,'omitnan');
    Q0_mean = mean(Q0_all,'omitnan');
    Q0_std = std(Q0_all,'omitnan');

    SolveSeconds_mean = mean(SolveSeconds_all);
    SolveSeconds_std = std(SolveSeconds_all);

    stats = [TP_all, FP_all, FN_all, Recall_all, Precision_all, ...
             F1_all, Error_all, SolveSeconds_all];

    % 保留 car100.m 的通用 stats/mean/sd，同时加入 KB100 风格命名。
    results(g).N = N;
    results(g).K = K;
    results(g).T = T;
    results(g).ompTau = ompTau;
    results(g).columns = metricNames;
    results(g).stats = stats;
    results(g).mean = mean(stats, 1);
    results(g).sd = std(stats, 0, 1);

    results(g).TP_all = TP_all;
    results(g).FP_all = FP_all;
    results(g).FN_all = FN_all;
    results(g).Recall_all = Recall_all;
    results(g).Precision_all = Precision_all;
    results(g).F1_all = F1_all;
    results(g).Error_all = Error_all;
    results(g).SolveSeconds_all = SolveSeconds_all;

    results(g).Recall_mean = Recall_mean;
    results(g).Recall_std = Recall_std;
    results(g).Precision_mean = Precision_mean;
    results(g).Precision_std = Precision_std;
    results(g).F1_mean = F1_mean;
    results(g).F1_std = F1_std;
    results(g).Error_mean = Error_mean;
    results(g).Error_std = Error_std;
    results(g).FullDetection_count = FullDetection_count;
    results(g).Q0_all = Q0_all;
    results(g).Q0Ratio_all = Q0Ratio_all;
    results(g).Q0Ratio_mean = Q0Ratio_mean;
    results(g).Q0Ratio_std = Q0Ratio_std;
    results(g).Q0_mean = Q0_mean;
    results(g).Q0_std = Q0_std;
    results(g).Q0Condition_all = Q0Condition_all;
    results(g).Q0_definition = 'half minimum outlier-pair distance over all sample local spaces';
    results(g).SolveSeconds_mean = SolveSeconds_mean;
    results(g).SolveSeconds_std = SolveSeconds_std;

    results(g).h = h;
    results(g).support_radius = support_radius;
    results(g).neighborCount = neighborCount;
    results(g).localRadii = localRadii;
    results(g).tA = tA;
    results(g).lastTrueIdx = index;
    results(g).lastDetectedIdx = ep_indices;
    results(g).lastEstimatedE = Ep;
    results(g).lastApproximation = f_approx;
    results(g).lastOMPIndex = Index;

    fprintf('\n==============================\n');
    fprintf('N = %d, K = %d, Trials = %d, OMP tau = %.2f\n', N, K, T, ompTau);
    fprintf('Recall    = %.6f +/- %.6f\n', Recall_mean, Recall_std);
    fprintf('Precision = %.6f +/- %.6f\n', Precision_mean, Precision_std);
    fprintf('F1        = %.6f +/- %.6f\n', F1_mean, F1_std);
    fprintf('Error     = %.6e +/- %.6e\n', Error_mean, Error_std);
    fprintf('Max local radius     = %.6e\n', support_radius);
    fprintf('omega_delta          = %.6e\n', omega_delta);
    fprintf('q_tilde_0            = %.6e +/- %.6e\n', Q0_mean, Q0_std);
    fprintf('q_tilde_0 / (max(delta_j)+omega_delta) = %.6e +/- %.6e\n',Q0Ratio_mean,Q0Ratio_std);
    fprintf('max(delta_j) + omega_delta = %.6e\n', support_radius + omega_delta);
    fprintf('Separation condition = %d / %d trials\n', ...
        sum(Q0Condition_all), T);
    fprintf('Full detection count = %d / %d trials\n', ...
        FullDetection_count, T);
    fprintf('A time          = %.6e s\n', tA);
    fprintf('Mean OMP time   = %.6e +/- %.6e s\n', ...
        SolveSeconds_mean, SolveSeconds_std);
    fprintf('==============================\n');

    % 每跑完一组 K 都保存，避免中断后丢失已完成的结果。
    save(resultFile, 'results', 'A', 'f_true', 'selectedFrameNumbers', 'eligibleIdx', ...
        'tA', 'h', 'support_radius', 'neighborCount', 'localRadii', ...
        'KList', 'T', 'useAllImages', 'geometry');
end

%% 6. 总耗时
trialSec = toc(trialClock);
totalSec = toc(overallClock);
fprintf('\n全部试验完成。A: %.2f s；随机实验: %.2f s；全程: %.2f s。\n', ...
    tA, trialSec, totalSec);
fprintf('结果已保存到：%s\n', resultFile);
