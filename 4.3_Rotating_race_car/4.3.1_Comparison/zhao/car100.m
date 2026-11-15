% Zhao_car100.m
% Rotating-race-car experiment: Zhao detection, 100 randomized trials.
% Same image selection, angle corruption, K values, and detection metrics as car100.
% Zhao does not estimate the sparse angular error E, so Error and A time are N/A.
% Saves every 10 trials. Incompatible checkpoints are backed up before restarting.

clearvars; clc;

%% 0. Settings
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end

imageFolder = 'E:\桌面\论文实验代码\MMLS函数\car';
useAllImages = false;                % false: central 60; true: all 72
centerCount = 60;                    % eligible positions for outliers
KList = [4, 8, 12];
T = 100;
kNN = 15;                            % Original Zhao implementation
rngSeed = 1;
hManual = [];                        % []: same max-nearest-neighbor estimate as car100
resetRun = false;                    % true: overwrite old checkpoint and run again

resultMat = fullfile(scriptDir, 'Zhao_car100_results.mat');
resultCsv = fullfile(scriptDir, 'Zhao_car100_summary.csv');

%% 1. Load images and clean angles
imageFiles = dir(fullfile(imageFolder, '*.png'));
totalImages = numel(imageFiles);
if totalImages < centerCount
    error('Need at least %d PNG images; found %d.', centerCount, totalImages);
end

% As in car100, compute angles from the full 72-image sequence first.
% Ensure filename order agrees with the physical rotation order.
allAngles = (0:totalImages-1)' * (2*pi/totalImages);
startIdx = floor((totalImages-centerCount)/2) + 1;
centerIdx = (startIdx:startIdx+centerCount-1)';

if useAllImages
    selectedFiles = imageFiles;
    fClean = allAngles;
    eligibleIdx = centerIdx;
else
    selectedFiles = imageFiles(centerIdx);
    fClean = allAngles(centerIdx);
    eligibleIdx = (1:centerCount)';
end
N = numel(selectedFiles);

if kNN < 1 || kNN >= N || kNN ~= floor(kNN)
    error('kNN must be an integer between 1 and N-1.');
end
if T < 1 || T ~= floor(T) || any(KList < 1) || ...
        any(KList ~= floor(KList)) || any(KList > numel(eligibleIdx))
    error('Invalid T or KList.');
end

firstImg = im2double(imread(fullfile(imageFolder, selectedFiles(1).name)));
if ~ismatrix(firstImg)
    error('Expected grayscale images.');
end
r = zeros(numel(firstImg), N);
r(:,1) = firstImg(:);
for i = 2:N
    img = im2double(imread(fullfile(imageFolder, selectedFiles(i).name)));
    if ~ismatrix(img) || ~isequal(size(img), size(firstImg))
        error('Image %s has inconsistent size or is not grayscale.', selectedFiles(i).name);
    end
    r(:,i) = img(:);
end
fprintf('Zhao: N=%d, image dimension=%d, eligible=%d, kNN=%d\n', ...
        N, size(r,1), numel(eligibleIdx), kNN);

%% 2. Reference h, support radius, and image-space distances
% h and 4*h are used ONLY for the shared spacing statistics.
% They are NOT parameters of the Zhao detector.
squaredNorms = sum(r.^2, 1);
D2 = bsxfun(@plus, squaredNorms(:), squaredNorms(:)') - 2*(r'*r);
D2 = max(D2, 0);
D2(1:N+1:end) = inf;
D = sqrt(D2);

if isempty(hManual)
    h = max(min(D, [], 2));
else
    h = hManual;
end
if ~isscalar(h) || ~isfinite(h) || h <= 0
    error('h must be a positive finite scalar.');
end
supportRadius = 4*h;
fprintf('h=%.6e, reference support radius=%.6e\n', h, supportRadius);

%% 3. Generate all randomized corruptions once (same RNG convention as car100)
rng(rngSeed, 'twister');
nK = numel(KList);
trueIdxBank = cell(T, nK);
Ebank = zeros(N, T, nK);
for g = 1:nK
    K = KList(g);
    for trial = 1:T
        trueIdx = sort(eligibleIdx(randperm(numel(eligibleIdx), K)));
        E = zeros(N, 1);
        E(trueIdx) = 5 + 5*rand(K, 1);
        trueIdxBank{trial,g} = trueIdx;
        Ebank(:,trial,g) = E;
    end
end

%% 4. Prepare checkpoints
metadata.N = N;
metadata.KList = KList;
metadata.T = T;
metadata.kNN = kNN;
metadata.rngSeed = rngSeed;
metadata.useAllImages = useAllImages;
metadata.imageFiles = {selectedFiles.name};
metadata.h = h;

resumePrevious = false;
if ~resetRun && isfile(resultMat)
    previous = load(resultMat, 'results', 'metadata');
    resumePrevious = isfield(previous,'metadata') && isfield(previous,'results') && ...
        isequaln(previous.metadata, metadata);
    if ~resumePrevious
        [~,backupToken] = fileparts(tempname(scriptDir));
        backupMat = fullfile(scriptDir, ...
            ['Zhao_car100_results_backup_' backupToken '.mat']);
        copyfile(resultMat, backupMat);
        if isfile(resultCsv)
            backupCsv = fullfile(scriptDir, ...
                ['Zhao_car100_summary_backup_' backupToken '.csv']);
            copyfile(resultCsv, backupCsv);
        end
        fprintf('Checkpoint settings differ. Previous results backed up to:\n%s\n',backupMat);
        fprintf('Starting a new run with the current settings.\n');
    end
end
if resumePrevious
    results = previous.results;
    fprintf('Existing checkpoint found; completed trials will be skipped.\n');
else
    results = struct([]);
    for g = 1:nK
        results(g).K = KList(g);
        % TP FP FN Recall Precision F1 MinSpacing SpacingRatio ZhaoSeconds
        results(g).stats = NaN(T,9);
        results(g).lastTrueIdx = [];
        results(g).lastDetectedIdx = [];
    end
end

%% 5. Repeat Zhao detector 
Xnoisy = [r; fClean'];                % (pixels+angle) x N
for g = 1:nK
    K = KList(g);
    for trial = 1:T
        if isfinite(results(g).stats(trial,1))
            continue;
        end
        trueIdx = trueIdxBank{trial,g};
        E = Ebank(:,trial,g);
        fNoisy = fClean + E;
        Xnoisy(end,:) = fNoisy';

        tStart = tic;
        [~, detected_outliers] = ZhaoMOD_original(Xnoisy, kNN);
        zhaoSeconds = toc(tStart);

        detectedIdx = unique(detected_outliers(:));
        if any(detectedIdx < 1) || any(detectedIdx > N) || ...
                any(detectedIdx ~= floor(detectedIdx))
            error('Zhao detector returned invalid sample indices.');
        end

        TP = numel(intersect(trueIdx, detectedIdx));
        FP = numel(setdiff(detectedIdx, trueIdx));
        FN = K - TP;
        recall = TP/K;
        precision = TP/max(TP+FP, 1);
        F1 = 2*TP/max(2*TP+FP+FN, 1);

        % Pairwise image-space distance: diagonal in D is already Inf.
        if K >= 2
            subD = D(trueIdx, trueIdx);
            minSpacing = min(subD(:));
            spacingRatio = minSpacing/supportRadius;
        else
            minSpacing = NaN;
            spacingRatio = NaN;
        end

        results(g).stats(trial,:) = [TP, FP, FN, recall, precision, ...
                                      F1, minSpacing, spacingRatio, zhaoSeconds];
        results(g).lastTrueIdx = trueIdx;
        results(g).lastDetectedIdx = detectedIdx;

        if trial == 1 || mod(trial,10) == 0 || trial == T
            fprintf('K=%d: %d/%d, TP=%d, FP=%d, F1=%.4f, Zhao time=%.2f s\n', ...
                    K, trial, T, TP, FP, F1, zhaoSeconds);
        end
        if mod(trial,10) == 0 || trial == T
            save(resultMat, 'results', 'metadata');
        end
    end

    s = results(g).stats;
    fprintf('\n============================================\n');
    fprintf('Zhao | N=%d, K=%d, Trials=%d\n', N, K, T);
    fprintf('TP        = %.6f +/- %.6f\n', mean(s(:,1)), std(s(:,1)));
    fprintf('FP        = %.6f +/- %.6f\n', mean(s(:,2)), std(s(:,2)));
    fprintf('FN        = %.6f +/- %.6f\n', mean(s(:,3)), std(s(:,3)));
    fprintf('Recall    = %.6f +/- %.6f\n', mean(s(:,4)), std(s(:,4)));
    fprintf('Precision = %.6f +/- %.6f\n', mean(s(:,5)), std(s(:,5)));
    fprintf('F1        = %.6f +/- %.6f\n', mean(s(:,6)), std(s(:,6)));
    fprintf('Error     = N/A (no sparse angular-error estimate)\n');
    fprintf('Support radius (reference) = %.6e\n', supportRadius);
    fprintf('Min outlier spacing = %.6e +/- %.6e\n', mean(s(:,7)), std(s(:,7)));
    fprintf('Min spacing / radius = %.6f +/- %.6f\n', mean(s(:,8)), std(s(:,8)));
    fprintf('A time = N/A (Zhao does not construct an MMLS A)\n');
    fprintf('Mean Zhao time = %.6f +/- %.6f s\n', mean(s(:,9)), std(s(:,9)));
    fprintf('FP >= 10 rate = %.4f\n', mean(s(:,2) >= 10));
    allDetectedCount = sum(s(:,1) == K);
    fprintf('All true outliers detected = %d/%d\n', allDetectedCount, T);
    fprintf('============================================\n');
end

%% 6. Write one-row-per-K CSV for the paper
summaryRows = struct([]);
for g = 1:nK
    s = results(g).stats;
    if any(~isfinite(s(:,1))), continue; end
    row = struct();
    row.Method = 'Zhao';
    row.N = N; row.K = KList(g); row.Trials = T;
    row.TP_mean = mean(s(:,1)); row.TP_std = std(s(:,1));
    row.FP_mean = mean(s(:,2)); row.FP_std = std(s(:,2));
    row.FN_mean = mean(s(:,3)); row.FN_std = std(s(:,3));
    row.Recall_mean = mean(s(:,4)); row.Recall_std = std(s(:,4));
    row.Precision_mean = mean(s(:,5)); row.Precision_std = std(s(:,5));
    row.F1_mean = mean(s(:,6)); row.F1_std = std(s(:,6));
    row.Error_mean = NaN; row.Error_std = NaN;
    row.SupportRadius = supportRadius;
    row.MinSpacing_mean = mean(s(:,7)); row.MinSpacing_std = std(s(:,7));
    row.SpacingRatio_mean = mean(s(:,8)); row.SpacingRatio_std = std(s(:,8));
    row.ATime_sec = NaN;
    row.ZhaoTime_mean_sec = mean(s(:,9));
    row.ZhaoTime_std_sec = std(s(:,9));
    row.FP_ge10_rate = mean(s(:,2) >= 10);
    summaryRows(end+1) = row; %#ok<SAGROW>
end

summaryTable = struct2table(summaryRows);
writetable(summaryTable, resultCsv);
save(resultMat, 'results', 'metadata', 'summaryTable');
fprintf('\nSaved: %s\nSaved: %s\n', resultMat, resultCsv);

%% 7. Original uploaded Zhao implementation (function name aligned locally)
% The calculations below are copied from MODbig.m without altering its
% detection rule. Localizing avoids the MOD / MODbig filename mismatch.

function [X_denoised, outliers] = ZhaoMOD_original(X, k)
% 流形去噪与异常值检测算法
% 输入:
%   X: 原始数据矩阵 (D×N)
%   k: 近邻数量
% 输出:
%   X_denoised: 去噪后的数据
%   outliers: 异常值索引

[D, N] = size(X);
X_denoised = zeros(D, N);
S_vectors = zeros(D, N);

for i = 1:N
    % 找到k个最近邻
    distances = sqrt(sum((X - X(:, i)).^2, 1));
    [~, idx] = sort(distances);
    neighbors = X(:, idx(2:k+1)); 

    % 使用SVD拟合局部流形（避免构造D×D矩阵）
    mu = mean(neighbors, 2);
    centered_neighbors = neighbors - mu;
    [U, ~, ~] = svd(centered_neighbors, 'econ');
    S = U(:, end);  % 最小奇异值方向
    S_vectors(:, i) = S;

    % 投影去噪
    d = abs(S' * (X(:, i) - mu));
    X_denoised(:, i) = X(:, i) - d * S;
end

% 噪声估计
delta = X - X_denoised;
delta_projected = sum(S_vectors .* delta, 1);

mu_noise = mean(delta_projected);
sigma_noise = std(delta_projected);
P = mean(delta_projected <= mu_noise);
sigma_l = P * sigma_noise;
sigma_u = (1 - P) * sigma_noise;

% 异常值检测
outliers = [];
for i = 1:N
    projection_value = S_vectors(:, i)' * X(:, i);
    denoised_projection = S_vectors(:, i)' * X_denoised(:, i);
    lower_bound = denoised_projection + mu_noise - 3 * sigma_l;
    upper_bound = denoised_projection + mu_noise + 3 * sigma_u;
    if projection_value < lower_bound || projection_value > upper_bound
        outliers = [outliers, i];
    end
end
end
