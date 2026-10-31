% car100_KBmetrics.m
% 旋转赛车图像实验：仿照 KB100.m 统计各项指标。
% A 仅构造一次；针对 K = 4, 8, 12 各做 100 次随机异常实验。
% 保留赛车实验的正向异常幅度 [5, 10]，不额外加入小噪声。

clearvars;
clc;
overallClock = tic;

%% =========================================================
% 0. 参数和函数路径
% ==========================================================
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end
addpath(scriptDir);

functionDir = fullfile(scriptDir, 'function');
if isfolder(functionDir)
    addpath(functionDir);
end

imageFolder = 'E:\桌面\论文实验代码\MMLS函数\car';

% false：原 car.m 的设置，直接使用中间 60 张图；
% true ：使用全部图（通常为 72 张），只在中间 60 张加入异常值。
% 若要复现论文第 4.3 节的实验设置，应改为 true。
useAllImages = false;
numCenterImages = 60;

epsilon = 1e-6;
d = 1;
m = 3;
hManual = [];  % []：以最大最近邻距离估计 h；也可指定已有的 h。

KList = [4, 8, 12];
T = 100;

if ~isscalar(T) || T < 1 || T ~= floor(T)
    error('T 必须是正整数。');
end
if isempty(KList) || any(KList < 1) || any(KList ~= floor(KList))
    error('KList 必须包含正整数。');
end

%% =========================================================
% 1. 加载图像和真实角度（只读取一次）
% ==========================================================
if ~isfolder(imageFolder)
    error('图像文件夹不存在：%s', imageFolder);
end
imageFiles = dir(fullfile(imageFolder, '*.png'));
totalImages = numel(imageFiles);

if totalImages < 2
    error('至少需要 2 张 PNG 图像。');
end

% 与原 car.m 一致：按照 dir 返回的文件顺序设置角度。
% 请确认文件名排序就是物体的旋转顺序。
allAngles = (0:totalImages-1)' * (2*pi/totalImages);

numCenterImages = min(numCenterImages, totalImages);
startIdx = floor((totalImages - numCenterImages)/2) + 1;
centerGlobalIdx = startIdx:(startIdx + numCenterImages - 1);

if useAllImages
    selectedFiles = imageFiles;
    f_true = allAngles;
    eligibleIdx = centerGlobalIdx(:);
else
    selectedFiles = imageFiles(centerGlobalIdx);
    f_true = allAngles(centerGlobalIdx);
    eligibleIdx = (1:numel(selectedFiles))';
end

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

%% =========================================================
% 2. 计算图像距离矩阵、h、支撑半径
% ==========================================================
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

% 与 FLCS.m / MMLSforA.m 中的 w_test(...,4,h) 对应。
support_radius = 6*h;
delta=support_radius ;
fprintf('h = %.6e, support radius = %.6e\n', h, support_radius);

%% =========================================================
% 3. 构造 A：只执行一次
% ==========================================================
fprintf('开始构造 MMLS 矩阵 A ...\n');
drawnow;

buildClock = tic;
A = MMLSforA(r, epsilon, d, m, h,delta);
tA = toc(buildClock);

if ~isequal(size(A), [N, N]) || any(~isfinite(A(:)))
    error('MMLSforA 返回的 A 尺寸不正确或存在 NaN/Inf。');
end

fprintf('A 构造完成：tA = %.6e s\n', tA);

% 后面只需要小型的图像间距矩阵，不再需要原始高维图像。
clear r D2 squareNorms firstImage img;

%% =========================================================
% 4. 随机异常实验：每个 K 做 T 次
% ==========================================================
% 列依次为：
% TP, FP, FN, Recall, Precision, F1, Error,
% MinSpacing, SpacingRatio, SolveSeconds
metricNames = {'TP','FP','FN','Recall','Precision','F1', ...
               'Error','MinSpacing','SpacingRatio','SolveSeconds'};

results = struct([]);
resultFile = fullfile(scriptDir, 'car_results_split_k=6.mat');

% 放在 A 构造之后，避免 svds 等操作影响异常值的随机序列。
rng(1);

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

    MinSpacing_all = nan(T,1);
    SpacingRatio_all = nan(T,1);
    SolveSeconds_all = zeros(T,1);

    for trial = 1:T
        % 每次试验都从干净的角度值重新开始。
        E = zeros(N,1);
        index = sort(eligibleIdx(randperm(numel(eligibleIdx), K)));

        % KB100 的最小异常点间距统计：
        % 图像点间距在原始像素向量的欧氏空间中计算。
        if K >= 2
            selectedDist = pairwiseDist(index, index);
            upperMask = triu(true(K), 1);
            min_spacing = min(selectedDist(upperMask));

            MinSpacing_all(trial) = min_spacing;
            SpacingRatio_all(trial) = min_spacing / support_radius;
        end
        % 当 K=1 时两两最小间距没有定义，保留 NaN。

        % 赛车实验：异常角度幅度为 [5,10]，保持原来的正向扰动。
        E(index) = 5 + 5*rand(K,1);
        f = f_true + E;

        % 只重复 OMP 与函数拟合，不重新构造 A。
        solveClock = tic;
        [Ep, Index, f_approx] = MMLSforoutlier(A, f);
        SolveSeconds_all(trial) = toc(solveClock);

        % 与 KB100.m 一致：根据估计异常向量判断检测位置。
        ep_indices = find(abs(Ep(:)) > 1e-12);

        TP = numel(intersect(index, ep_indices));
        FP = numel(setdiff(ep_indices, index));
        FN = numel(setdiff(index, ep_indices));

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

        % 与 KB100.m 一致：分母使用真实异常点数量 K。
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

    MinSpacing_mean = mean(MinSpacing_all);
    MinSpacing_std = std(MinSpacing_all);

    SpacingRatio_mean = mean(SpacingRatio_all);
    SpacingRatio_std = std(SpacingRatio_all);

    SolveSeconds_mean = mean(SolveSeconds_all);
    SolveSeconds_std = std(SolveSeconds_all);

    stats = [TP_all, FP_all, FN_all, Recall_all, Precision_all, ...
             F1_all, Error_all, MinSpacing_all, SpacingRatio_all, ...
             SolveSeconds_all];

    % 保留 car100.m 的通用 stats/mean/sd，同时加入 KB100 风格命名。
    results(g).N = N;
    results(g).K = K;
    results(g).T = T;
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
    results(g).MinSpacing_all = MinSpacing_all;
    results(g).SpacingRatio_all = SpacingRatio_all;
    results(g).SolveSeconds_all = SolveSeconds_all;

    results(g).Recall_mean = Recall_mean;
    results(g).Recall_std = Recall_std;
    results(g).Precision_mean = Precision_mean;
    results(g).Precision_std = Precision_std;
    results(g).F1_mean = F1_mean;
    results(g).F1_std = F1_std;
    results(g).Error_mean = Error_mean;
    results(g).Error_std = Error_std;
    results(g).MinSpacing_mean = MinSpacing_mean;
    results(g).MinSpacing_std = MinSpacing_std;
    results(g).SpacingRatio_mean = SpacingRatio_mean;
    results(g).SpacingRatio_std = SpacingRatio_std;
    results(g).SolveSeconds_mean = SolveSeconds_mean;
    results(g).SolveSeconds_std = SolveSeconds_std;

    results(g).h = h;
    results(g).support_radius = support_radius;
    results(g).tA = tA;
    results(g).lastTrueIdx = index;
    results(g).lastDetectedIdx = ep_indices;
    results(g).lastEstimatedE = Ep;
    results(g).lastApproximation = f_approx;
    results(g).lastOMPIndex = Index;

    fprintf('\n==============================\n');
    fprintf('N = %d, K = %d, Trials = %d\n', N, K, T);
    fprintf('TP        = %.6f +/- %.6f\n', mean(TP_all), std(TP_all));
    fprintf('FP        = %.6f +/- %.6f\n', mean(FP_all), std(FP_all));
    fprintf('FN        = %.6f +/- %.6f\n', mean(FN_all), std(FN_all));
    fprintf('Recall    = %.6f +/- %.6f\n', Recall_mean, Recall_std);
    fprintf('Precision = %.6f +/- %.6f\n', Precision_mean, Precision_std);
    fprintf('F1        = %.6f +/- %.6f\n', F1_mean, F1_std);
    fprintf('Error     = %.6e +/- %.6e\n', Error_mean, Error_std);
    fprintf('Support radius       = %.6e\n', support_radius);
    fprintf('Min outlier spacing  = %.6e +/- %.6e\n', ...
        MinSpacing_mean, MinSpacing_std);
    fprintf('Min spacing / radius = %.6f +/- %.6f\n', ...
        SpacingRatio_mean, SpacingRatio_std);
    fprintf('A time          = %.6e s\n', tA);
    fprintf('Mean OMP time   = %.6e +/- %.6e s\n', ...
        SolveSeconds_mean, SolveSeconds_std);
    fprintf('==============================\n');

    % 每跑完一组 K 都保存，避免中断后丢失已完成的结果。
    save(resultFile, 'results', 'A', 'f_true', 'eligibleIdx', ...
        'tA', 'h', 'support_radius', 'KList', 'T', 'useAllImages');
end

%% =========================================================
% 6. 总耗时
% ==========================================================
trialSec = toc(trialClock);
totalSec = toc(overallClock);
fprintf('\n全部试验完成。A: %.2f s；随机实验: %.2f s；全程: %.2f s。\n', ...
    tA, trialSec, totalSec);
fprintf('结果已保存到：%s\n', resultFile);
