% Race car time and memory benchmark (Our method, K=8).
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

d = 1;
m = 3;
hManual = [];  % []：以最大最近邻距离估计 h；也可指定已有的 h。

KList = [8];
T = 100;
ompTau = 0.5;
outlierMagnitude = [5,10];
referenceKList = [4,8,12];  % car100 依次生成各组的异常数据。
referenceTrials = 100;
profileK = 8;          % 仅 K=8 记录额外时间/内存
benchmarkK8 = struct(); % 本次运行新生成，不读取旧的检查点

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

% 为公平比较，同时记录 MMLS 的一次高维预计算开销。
beforeA = MMLSMemorySnapshot();
monitorA = MMLSStartMemoryMonitor('GROUP');
monitorACleanup = onCleanup(@() MMLSCleanupMemoryMonitor(monitorA));
buildClock = tic;
% 与 car100 一致，对原始图像向量采用查询点固定邻域。
[A,neighborCount,localRadii,geometry] = buildLocalPcaMLS(r.',d,m,3,true);
tA = toc(buildClock);
afterA = MMLSMemorySnapshot();
MMLSSetMonitorPhase(monitorA, 'STOP');
[~, peakA, ~, samplesA] = MMLSReadMonitor(monitorA, 10);
clear monitorACleanup;

support_radius = max(localRadii);
if ~isequal(size(A), [N, N]) || any(~isfinite(A(:)))
    error('buildLocalPcaMLS 返回的 A 尺寸不正确或存在 NaN/Inf。');
end

AStorage = whos('A');
AStorageBytes = AStorage.bytes;
AStorageMiB = AStorageBytes/(1024^2);
memoryA = struct( ...
    'ATime_sec', tA, ...
    'Storage_bytes', AStorageBytes, ...
    'Storage_MiB', AStorageMiB, ...
    'MemBefore_MiB', beforeA.matlabMiB, ...
    'MemAfter_MiB', afterA.matlabMiB, ...
    'MemChange_MiB', afterA.matlabMiB-beforeA.matlabMiB, ...
    'WorkingSetBefore_MiB', beforeA.workingSetMiB, ...
    'WorkingSetAfter_MiB', afterA.workingSetMiB, ...
    'PeakWorkingSet_MiB', peakA, ...
    'PeakIncrease_MiB', MMLSPeakIncrease(peakA,beforeA.workingSetMiB), ...
    'PeakSamples', samplesA);
fprintf('A 构造完成：tA = %.6e s\n', tA);
fprintf('Neighbors per row = %d to %d, search radius sigma = %.6e\n', ...
    min(neighborCount),max(neighborCount),geometry.sigma);
fprintf('矩阵 A 自身存储内存 = %.6f MiB（%d bytes）\n',AStorageMiB,AStorageBytes);
fprintf('A 峰值工作集（采样）= %.2f MiB，采样次数 = %d\n',peakA,samplesA);

% 后面只需要小型的图像间距矩阵，不再需要原始高维图像。
clear r D2 squareNorms firstImage img;

%% 4. 随机异常实验：每个 K 做 T 次
% 列依次为：
% TP, FP, FN, Recall, Precision, F1, Error,
% MinSpacing, SpacingRatio, SolveSeconds
metricNames = {'TP','FP','FN','Recall','Precision','F1', ...
               'Error','MinSpacing','SpacingRatio','SolveSeconds'};

results = struct([]);
% 使用新的保存路径，避免覆盖原有 car100 实验结果。
resultFile = fullfile(scriptDir, 'car_results_MMLS_K8_time_memory.mat');
profileCsv = fullfile(scriptDir, 'MMLS_K8_time_memory.csv');

trialClock = tic;

for g = 1:numel(KList)
    K = KList(g);
    outlierStream = RandStream('mt19937ar','Seed',1);
    referenceGroup = find(referenceKList==K,1);
    if ~isempty(referenceGroup)
        % 重放 car100 中前面各组的抽样，使 K=8 的首轮及后续轮次逐一配对。
        for previousK = referenceKList(1:(referenceGroup-1))
            for trial = 1:referenceTrials
                MMLSSignedOutliers(outlierStream,numel(eligibleIdx),previousK,outlierMagnitude);
            end
        end
    end

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

    % K=8 才启动组内连续监控，K=4、12 的原有统计不变。
    isProfile = (K == profileK);
    if isProfile
        before100 = MMLSMemorySnapshot();
        monitor8 = MMLSStartMemoryMonitor('SINGLE');
        monitor8Cleanup = onCleanup(@() MMLSCleanupMemoryMonitor(monitor8));
        groupClock = tic;
        fprintf('\n========== K=8：开始单次与100次计时、内存测量 ==========\n');
    end

    for trial = 1:T
        [eligibleOutlierIndex,eligibleOutliers] = MMLSSignedOutliers( ...
            outlierStream,numel(eligibleIdx),K,outlierMagnitude);
        index = eligibleIdx(eligibleOutlierIndex);
        E = zeros(N,1);
        E(eligibleIdx) = eligibleOutliers;

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

        f = f_true + E;

        % 只重复 OMP 与函数拟合，不重新构造 A。
        if isProfile && trial == 1
            beforeOne = MMLSMemorySnapshot();
        end

        solveClock = tic;
        [Ep, Index, f_approx] = MMLSforoutlier(A, f, [], [], ompTau);
        SolveSeconds_all(trial) = toc(solveClock);

        if isProfile && trial == 1
            afterOne = MMLSMemorySnapshot();
            MMLSSetMonitorPhase(monitor8,'GROUP');
            fprintf('K=8 首次 OMP 耗时：%.6f s\n',SolveSeconds_all(1));
            MMLSPrintSnapshot('单次OMP',beforeOne,afterOne);
        end

        ep_indices = find(Ep(:) ~= 0);

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

        Error_all(trial) = norm(Ep(:) - E, 2) / K;

        if mod(trial, 10) == 0 || trial == 1
            fprintf('K=%d：第 %d/%d 次完成，总用时 %.1f s\n', ...
                K, trial, T, toc(trialClock));
        end
    end

    [maxSolveSeconds,maxSolveTrial] = max(SolveSeconds_all);
    if isProfile
        % 该时间包含100次异常值生成、指标计算和显示，但不含 A 预计算。
        groupWallSeconds = toc(groupClock);
        after100 = MMLSMemorySnapshot();
        MMLSSetMonitorPhase(monitor8,'STOP');
        [peakOne, peak100, samplesOne, samples100] = MMLSReadMonitor(monitor8,10);
        clear monitor8Cleanup;

        singleTime = SolveSeconds_all(1);
        sum100Time = sum(SolveSeconds_all);  % 100次真实检测耗时之和，不是估计
        benchmarkK8 = struct( ...
            'N',N, 'K',K, 'Trials',T, 'OMPTau',ompTau, ...
            'ATime_sec',tA, ...
            'SingleOMPTime_sec',singleTime, ...
            'Total100OMPTime_sec',sum100Time, ...
            'Total100WallTime_sec',groupWallSeconds, ...
            'MeanOMPTime_sec',mean(SolveSeconds_all), ...
            'StdOMPTime_sec',std(SolveSeconds_all), ...
            'MaxOMPTime_sec',maxSolveSeconds, ...
            'MaxOMPTimeTrial',maxSolveTrial, ...
            'A_Storage_bytes',AStorageBytes, ...
            'A_Storage_MiB',AStorageMiB, ...
            'APlusSingleTime_sec',tA+singleTime, ...
            'APlus100OMPTime_sec',tA+sum100Time, ...
            'APlus100WallTime_sec',tA+groupWallSeconds, ...
            'SingleMemBefore_MiB',beforeOne.matlabMiB, ...
            'SingleMemAfter_MiB',afterOne.matlabMiB, ...
            'SingleMemChange_MiB',afterOne.matlabMiB-beforeOne.matlabMiB, ...
            'SingleWorkingSetBefore_MiB',beforeOne.workingSetMiB, ...
            'SingleWorkingSetAfter_MiB',afterOne.workingSetMiB, ...
            'SinglePeakWorkingSet_MiB',peakOne, ...
            'SinglePeakIncrease_MiB',MMLSPeakIncrease(peakOne,beforeOne.workingSetMiB), ...
            'SinglePeakSamples',samplesOne, ...
            'AllMemBefore_MiB',before100.matlabMiB, ...
            'AllMemAfter_MiB',after100.matlabMiB, ...
            'AllMemChange_MiB',after100.matlabMiB-before100.matlabMiB, ...
            'AllWorkingSetBefore_MiB',before100.workingSetMiB, ...
            'AllWorkingSetAfter_MiB',after100.workingSetMiB, ...
            'AllPeakWorkingSet_MiB',peak100, ...
            'AllPeakIncrease_MiB',MMLSPeakIncrease(peak100,before100.workingSetMiB), ...
            'AllPeakSamples',samples100, ...
            'A_MemBefore_MiB',memoryA.MemBefore_MiB, ...
            'A_MemAfter_MiB',memoryA.MemAfter_MiB, ...
            'A_PeakWorkingSet_MiB',memoryA.PeakWorkingSet_MiB, ...
            'A_PeakIncrease_MiB',memoryA.PeakIncrease_MiB, ...
            'A_PeakSamples',memoryA.PeakSamples);

        fprintf('\n================ K=8：时间和内存 ================\n');
        fprintf('构造 A 时间                    : %.3f s\n',tA);
        fprintf('单次 MMLS/OMP 检测时间         : %.6f s\n',singleTime);
        fprintf('单次检测最大耗时               : %.6f s（第%d次）\n',maxSolveSeconds,maxSolveTrial);
        fprintf('矩阵 A 自身存储内存            : %.6f MiB（%d bytes）\n',AStorageMiB,AStorageBytes);
        fprintf('100次检测累计时间              : %.6f s\n',sum100Time);
        fprintf('100次循环实际时间              : %.6f s\n',groupWallSeconds);
        fprintf('构造 A + 100次检测累计时间     : %.3f s\n',tA+sum100Time);
        MMLSPrintSnapshot('单次OMP',beforeOne,afterOne);
        fprintf('单次 OMP 采样峰值工作集         : %.2f MiB（%d次采样）\n',peakOne,samplesOne);
        MMLSPrintSnapshot('100次OMP',before100,after100);
        fprintf('100次 OMP 采样峰值工作集        : %.2f MiB（%d次采样）\n',peak100,samples100);
        fprintf('构造 A 的采样峰值工作集         : %.2f MiB（%d次采样）\n',peakA,samplesA);
        fprintf('提示：100次内存不累加；OMP 耗时很短，单次采样可能为 NaN。\n');
        fprintf('===================================================\n');

        writetable(struct2table(benchmarkK8), profileCsv);
        fprintf('K=8 计时/内存结果已保存：%s\n',profileCsv);
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
    results(g).MaxOMPTime_sec = maxSolveSeconds;
    results(g).MaxOMPTimeTrial = maxSolveTrial;

    results(g).h = h;
    results(g).support_radius = support_radius;
    results(g).tA = tA;
    results(g).memoryA = memoryA;
    if isProfile
        results(g).benchmarkK8 = benchmarkK8;
    end
    results(g).lastTrueIdx = index;
    results(g).lastDetectedIdx = ep_indices;
    results(g).lastEstimatedE = Ep;
    results(g).lastApproximation = f_approx;
    results(g).lastOMPIndex = Index;

    fprintf('\n==============================\n');
    fprintf('N = %d, K = %d, Trials = %d, OMP tau = %.2f\n', N, K, T, ompTau);
    fprintf('TP        = %.6f +/- %.6f\n', mean(TP_all), std(TP_all));
    fprintf('FP        = %.6f +/- %.6f\n', mean(FP_all), std(FP_all));
    fprintf('FN        = %.6f +/- %.6f\n', mean(FN_all), std(FN_all));
    fprintf('Recall    = %.6f +/- %.6f\n', Recall_mean, Recall_std);
    fprintf('Precision = %.6f +/- %.6f\n', Precision_mean, Precision_std);
    fprintf('F1        = %.6f +/- %.6f\n', F1_mean, F1_std);
    fprintf('Error     = %.6e +/- %.6e\n', Error_mean, Error_std);
    fprintf('Max local radius     = %.6e\n', support_radius);
    fprintf('Min outlier spacing  = %.6e +/- %.6e\n', ...
        MinSpacing_mean, MinSpacing_std);
    fprintf('Min spacing / radius = %.6f +/- %.6f\n', ...
        SpacingRatio_mean, SpacingRatio_std);
    fprintf('A time          = %.6e s\n', tA);
    fprintf('Mean OMP time   = %.6e +/- %.6e s\n', ...
        SolveSeconds_mean, SolveSeconds_std);
    fprintf('Max OMP time    = %.6e s (trial %d / %d)\n', ...
        maxSolveSeconds,maxSolveTrial,T);
    fprintf('==============================\n');

    % 每跑完一组 K 都保存，避免中断后丢失已完成的结果。
    save(resultFile, 'results', 'A', 'f_true', 'selectedFrameNumbers', 'eligibleIdx', ...
        'tA', 'h', 'support_radius', 'localRadii', 'neighborCount', ...
        'KList', 'T', 'useAllImages', 'ompTau', 'memoryA', 'benchmarkK8');
end

%% 6. 总耗时
trialSec = toc(trialClock);
totalSec = toc(overallClock);
fprintf('\n全部试验完成。A: %.2f s；随机实验: %.2f s；全程: %.2f s。\n', ...
    tA, trialSec, totalSec);
fprintf('结果已保存到：%s\n', resultFile);

%% =========================================================
% 7. MATLAB / Windows 进程内存测量辅助函数（不改变算法）
% 说明：工作集为 MATLAB 进程常驻物理内存，不等于算法独占的内存。
% 外部 PowerShell 每约100ms采样，不保证捕获毫秒级OMP的瞬时峰值。
%% =========================================================

function s = MMLSMemorySnapshot()
s = struct('matlabMiB',NaN,'workingSetMiB',NaN);
if ~ispc, return; end

try
    m = memory;
    s.matlabMiB = double(m.MemUsedMATLAB) / (1024^2);
catch
    % 某些 MATLAB/系统版本不提供 MemUsedMATLAB。
end

try
    proc = System.Diagnostics.Process.GetCurrentProcess();
    proc.Refresh();
    s.workingSetMiB = double(proc.WorkingSet64) / (1024^2);
catch
    % 无法访问 .NET 时仅保留 memory 的结果。
end
end

function v = MMLSPeakIncrease(peak, initialValue)
if isfinite(peak) && isfinite(initialValue)
    v = max(0,peak-initialValue);
else
    v = NaN;
end
end

function MMLSPrintSnapshot(label,before,after)
fprintf('%s: MATLAB 内存 %.2f -> %.2f MiB（变化 %+.2f MiB）\n', ...
    label,before.matlabMiB,after.matlabMiB,after.matlabMiB-before.matlabMiB);
fprintf('%s: 物理工作集 %.2f -> %.2f MiB（变化 %+.2f MiB）\n', ...
    label,before.workingSetMiB,after.workingSetMiB, ...
    after.workingSetMiB-before.workingSetMiB);
end

function mon = MMLSStartMemoryMonitor(initialPhase)
% 外部进程监控，避免在阻塞SVD期间依赖MATLAB timer的执行。
% SINGLE：K=8 首轮；GROUP：整组和 A；STOP：结束采样。
mon = struct('active',false,'folder','','phaseFile','', ...
    'readyFile','','resultFile','');

if ~ispc
    warning('峰值内存采样仅支持 Windows；计时和原算法继续运行。');
    return;
end

try
    process = System.Diagnostics.Process.GetCurrentProcess();
    pid = double(process.Id);
    mon.folder = tempname(tempdir);
    mkdir(mon.folder);
    mon.phaseFile = fullfile(mon.folder,'phase.txt');
    mon.readyFile = fullfile(mon.folder,'ready.txt');
    mon.resultFile = fullfile(mon.folder,'result.txt');
    scriptFile = fullfile(mon.folder,'monitor.ps1');
    MMLSWriteText(mon.phaseFile,initialPhase);

    psLines = {
        'param([int]$TargetPid,[string]$PhaseFile,[string]$ReadyFile,[string]$ResultFile)'
        '$ErrorActionPreference = "Stop"'
        '$culture = [System.Globalization.CultureInfo]::InvariantCulture'
        '$peakSingle = [double]0; $peakAll = [double]0'
        '$nSingle = 0; $nAll = 0'
        '$lastPhase = ""'
        '$limit = (Get-Date).AddHours(12)'
        '[System.IO.File]::WriteAllText($ReadyFile,"ready")'
        'while ((Get-Date) -lt $limit) {'
        '  try { $phase = ([System.IO.File]::ReadAllText($PhaseFile)).Trim() } catch { $phase = $lastPhase }'
        '  if ($phase -eq "STOP") { break }'
        '  try { $proc = Get-Process -Id $TargetPid -ErrorAction Stop } catch { break }'
        '  $mb = [double]$proc.WorkingSet64 / 1MB'
        '  if ($phase -eq "SINGLE") {'
        '    if ($mb -gt $peakSingle) { $peakSingle = $mb }'
        '    $nSingle++'
        '  }'
        '  if ($phase -eq "SINGLE" -or $phase -eq "GROUP") {'
        '    if ($mb -gt $peakAll) { $peakAll = $mb }'
        '    $nAll++'
        '  }'
        '  $lastPhase = $phase'
        '  Start-Sleep -Milliseconds 100'
        '}'
        '[System.IO.File]::WriteAllLines($ResultFile,[string[]]@($peakSingle.ToString($culture),$peakAll.ToString($culture),$nSingle.ToString(),$nAll.ToString()))'
    };

    fid = fopen(scriptFile,'w');
    if fid < 0, error('无法创建 PowerShell 采样脚本。'); end
    for i=1:numel(psLines)
        fprintf(fid,'%s\n',psLines{i});
    end
    fclose(fid);

    pinfo = System.Diagnostics.ProcessStartInfo();
    pinfo.FileName = 'powershell.exe';
    pinfo.Arguments = sprintf([ ...
        '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass ' ...
        '-File "%s" -TargetPid %.0f -PhaseFile "%s" ' ...
        '-ReadyFile "%s" -ResultFile "%s"'], ...
        scriptFile,pid,mon.phaseFile,mon.readyFile,mon.resultFile);
    pinfo.UseShellExecute = false;
    pinfo.CreateNoWindow = true;
    child = System.Diagnostics.Process.Start(pinfo);

    for j = 1:60
        if isfile(mon.readyFile)
            mon.active = true;
            fprintf('Windows MATLAB 内存峰值采样已启动（约100ms/次）。\n');
            return;
        end
        if child.HasExited, break; end
        pause(0.1);
    end
    warning('PowerShell 采样未成功启动；峰值记录为 NaN。');
catch ME
    warning('内存峰值采样不可用：%s。继续计算，峰值显示 NaN。',ME.message);
end
end

function MMLSSetMonitorPhase(mon,phase)
if ~mon.active, return; end
try
    MMLSWriteText(mon.phaseFile,phase);
catch ME
    warning('更新内存采样状态失败：%s',ME.message);
end
end

function [peakSingle,peakAll,nSingle,nAll] = MMLSReadMonitor(mon,timeoutSeconds)
peakSingle = NaN;
peakAll = NaN;
nSingle = 0;
nAll = 0;
if ~mon.active, return; end

clock = tic;
while ~isfile(mon.resultFile) && toc(clock)<timeoutSeconds
    pause(0.1);
end
if ~isfile(mon.resultFile)
    warning('内存采样器未能输出统计结果。');
    return;
end

try
    vals = sscanf(fileread(mon.resultFile),'%f');
    if numel(vals)==4
        nSingle = vals(3);
        nAll = vals(4);
        if nSingle>0, peakSingle = vals(1); end
        if nAll>0, peakAll = vals(2); end
    else
        warning('内存峰值日志格式异常。');
    end
catch ME
    warning('读取内存峰值日志失败：%s',ME.message);
end
end

function MMLSWriteText(filename,txt)
fid = fopen(filename,'w');
if fid<0, error('无法写入内存监控控制文件：%s',filename); end
fprintf(fid,'%s',txt);
fclose(fid);
end

function MMLSCleanupMemoryMonitor(mon)
if isempty(mon.folder), return; end
try
    if mon.active && ~isfile(mon.resultFile)
        MMLSSetMonitorPhase(mon,'STOP');
        for j = 1:30
            if isfile(mon.resultFile), break; end
            pause(0.1);
        end
    end
    if isfile(mon.resultFile)
        pause(0.1);
        rmdir(mon.folder,'s');
    end
catch
end
end
