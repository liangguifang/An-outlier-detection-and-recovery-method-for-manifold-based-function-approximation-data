% K=8 Zhao 100-trial race-car experiment, with runtime/memory profiling.
% Independent fresh run: never LOADs old .mat results or checkpoints.
% Keeps the Zhao detector identical to the originally uploaded car100.m.
% Memory: snapshots + independently sampled Windows MATLAB working set.
% Report first detection and 100 detections separately.
%
% Run in MATLAB: car100_Zhao_K8_time_memory

clc;

%% 0. Settings
scriptDir = fileparts(mfilename('fullpath'));
imageFolder = 'E:\桌面\论文实验代码\MMLS函数\car';
useAllImages = false;           % false = middle 60, true = all 72
centerCount = 60;
K = 8;                          % Only K=8 in this benchmark
T = 100;
kNN = 15;
rngSeed = 1;
hManual = [];
matchOldK8RandomTrials = true;  % Skip K=4 RNG draws from earlier [4 8 12] script

resultMat = fullfile(scriptDir, 'Zhao_K8_100_fresh_results.mat');
summaryCsv = fullfile(scriptDir, 'Zhao_K8_100_fresh_summary.csv');
benchCsv = fullfile(scriptDir, 'Zhao_K8_100_time_memory.csv');

%% 1. Load the image inputs once. No old benchmark MAT file is read.
imageFiles = dir(fullfile(imageFolder,'*.png'));
totalImages = numel(imageFiles);
if totalImages < centerCount
    error('Not enough images. Found %d; need at least %d.', ...
          totalImages, centerCount);
end

allAngles = (0:totalImages-1)'*(2*pi/totalImages);
startIdx = floor((totalImages-centerCount)/2)+1;
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
    error('kNN must be an integer in [1,N-1].');
end
if K > numel(eligibleIdx)
    error('K exceeds number of eligible images.');
end

firstImg = im2double(imread(fullfile(imageFolder,selectedFiles(1).name)));
if ~ismatrix(firstImg)
    error('Images must be grayscale.');
end
r = zeros(numel(firstImg),N);
r(:,1) = firstImg(:);
for i = 2:N
    img = im2double(imread(fullfile(imageFolder,selectedFiles(i).name)));
    if ~ismatrix(img) || ~isequal(size(img), size(firstImg))
        error('Invalid image size or color format: %s',selectedFiles(i).name);
    end
    r(:,i) = img(:);
end
fprintf('Zhao: N=%d, image dimension=%d, K=%d, trials=%d, kNN=%d\n', ...
        N,size(r,1),K,T,kNN);

%% 2. Reference h and outlier spacings (same as earlier car100)
squareNorms = sum(r.^2,1);
D2 = bsxfun(@plus,squareNorms(:),squareNorms(:)') - 2*(r'*r);
D2 = max(D2,0);
D2(1:N+1:end) = inf;
D = sqrt(D2);

if isempty(hManual)
    h = max(min(D,[],2));
else
    h = hManual;
end
if ~isscalar(h) || ~isfinite(h) || h <= 0
    error('Invalid h parameter.');
end
supportRadius = 4*h;
fprintf('h=%.6e, reference support radius=%.6e\n',h,supportRadius);
clear D2 squareNorms firstImg img;

%% 3. Prepare randomized outliers (no experiments run for K=4/12)
rng(rngSeed,'twister');
if matchOldK8RandomTrials
    % Earlier Zhao_car100 script pre-generated 100 K=4 trials before K=8.
    % Consuming the same RNG sequence reproduces its K=8 trials.
    for j = 1:T
        randperm(numel(eligibleIdx),4);
        rand(4,1);
    end
end

trueIdxBank = cell(T,1);
Ebank = zeros(N,T);
for trial = 1:T
    idx = sort(eligibleIdx(randperm(numel(eligibleIdx),K)));
    E = zeros(N,1);
    E(idx) = 5+5*rand(K,1);
    trueIdxBank{trial} = idx;
    Ebank(:,trial) = E;
end

% No previous results are loaded; always run all 100 calls from scratch.
stats = NaN(T,9);   % TP FP FN Recall Precision F1 Spacing Ratio ZhaoTime
metadata = struct('N',N,'K',K,'T',T,'kNN',kNN,'rngSeed',rngSeed, ...
    'h',h,'supportRadius',supportRadius,'useAllImages',useAllImages, ...
    'matchOldK8RandomTrials',matchOldK8RandomTrials, ...
    'imageFiles',{ {selectedFiles.name} });

%% 4. Sample memory while MATLAB performs blocking SVD calls
Xnoisy = [r;fClean'];
before100 = zhaoMemorySnapshot();
monitor = zhaoStartMonitor();
cleanupMonitor = onCleanup(@() zhaoCleanupMonitor(monitor));
blockClock = tic;

for trial = 1:T
    trueIdx = trueIdxBank{trial};
    E = Ebank(:,trial);
    Xnoisy(end,:) = (fClean+E)';

    if trial == 1
        beforeOne = zhaoMemorySnapshot();
    end

    % Stopwatch surrounds ONLY the actual Zhao algorithm.
    algoClock = tic;
    [~, detected] = ZhaoMOD_original(Xnoisy,kNN);
    zhaoSeconds = toc(algoClock);

    if trial == 1
        afterOne = zhaoMemorySnapshot();
        zhaoSetPhase(monitor,'GROUP');
        fprintf('\n======= K=8: first detection =======\n');
        fprintf('Single Zhao time: %.3f seconds\n',zhaoSeconds);
        zhaoPrintSnapshot('single',beforeOne,afterOne);
        fprintf('Single peak will be reported after the 100-run block.\n');
    end

    detectedIdx = unique(detected(:));
    if any(detectedIdx < 1) || any(detectedIdx > N) || ...
            any(detectedIdx ~= floor(detectedIdx))
        error('Zhao returned invalid outlier index.');
    end

    TP = numel(intersect(trueIdx,detectedIdx));
    FP = numel(setdiff(detectedIdx,trueIdx));
    FN = K-TP;
    recall = TP/K;
    precision = TP/max(TP+FP,1);
    F1 = 2*TP/max(2*TP+FP+FN,1);

    subD = D(trueIdx,trueIdx);
    minSpacing = min(subD(:));
    spacingRatio = minSpacing/supportRadius;
    stats(trial,:) = [TP FP FN recall precision F1 ...
                      minSpacing spacingRatio zhaoSeconds];

    if trial == 1 || mod(trial,10)==0 || trial==T
        fprintf('K=8: %d/%d | TP=%d FP=%d F1=%.4f | %.3f s\n', ...
            trial,T,TP,FP,F1,zhaoSeconds);
    end

    % Write progress but do not read/resume any previous checkpoint.
    if mod(trial,10)==0
        save(resultMat,'stats','metadata','-v7.3');
    end
end

blockSeconds = toc(blockClock);
after100 = zhaoMemorySnapshot();
zhaoSetPhase(monitor,'STOP');
[peakOne,peak100,nOne,n100] = zhaoReadMonitor(monitor,8);
clear cleanupMonitor;

%% 5. Final time/memory report
timeOne = stats(1,9);
time100 = sum(stats(:,9));     % exactly 100 Zhao calls, not an estimate
meanZhaoTime = mean(stats(:,9));
stdZhaoTime = std(stats(:,9));
[maxZhaoTime,maxTimeTrial] = max(stats(:,9));

benchmarkK8 = struct( ...
    'N',N,'K',K,'Trials',T, ...
    'SingleTime_sec',timeOne, ...
    'TotalZhaoTime_sec',time100, ...
    'TotalWallTime_sec',blockSeconds, ...
    'MeanZhaoTime_sec',meanZhaoTime, ...
    'StdZhaoTime_sec',stdZhaoTime, ...
    'MaxZhaoTime_sec',maxZhaoTime, ...
    'MaxZhaoTimeTrial',maxTimeTrial, ...
    'SingleMATLABBefore_MiB',beforeOne.matlabMiB, ...
    'SingleMATLABAfter_MiB',afterOne.matlabMiB, ...
    'SingleMATLABChange_MiB',afterOne.matlabMiB-beforeOne.matlabMiB, ...
    'SingleWorkingSetBefore_MiB',beforeOne.workingSetMiB, ...
    'SingleWorkingSetAfter_MiB',afterOne.workingSetMiB, ...
    'SinglePeakWorkingSet_MiB',peakOne, ...
    'SinglePeakIncrease_MiB',zhaoPeakIncrease(peakOne,beforeOne.workingSetMiB), ...
    'AllMATLABBefore_MiB',before100.matlabMiB, ...
    'AllMATLABAfter_MiB',after100.matlabMiB, ...
    'AllMATLABChange_MiB',after100.matlabMiB-before100.matlabMiB, ...
    'AllWorkingSetBefore_MiB',before100.workingSetMiB, ...
    'AllWorkingSetAfter_MiB',after100.workingSetMiB, ...
    'AllPeakWorkingSet_MiB',peak100, ...
    'AllPeakIncrease_MiB',zhaoPeakIncrease(peak100,before100.workingSetMiB), ...
    'SinglePeakSamples',nOne, ...
    'AllPeakSamples',n100);

fprintf('\n========== K=8: TIME / MEMORY ==========\n');
fprintf('One Zhao detection time:       %.3f s\n',timeOne);
fprintf('100 Zhao detection time SUM:   %.3f s (%.2f min)\n', ...
        time100,time100/60);
fprintf('100-run wall time incl. I/O:  %.3f s (%.2f min)\n', ...
        blockSeconds,blockSeconds/60);
fprintf('Average Zhao detection time:  %.3f +/- %.3f s\n', ...
        meanZhaoTime,stdZhaoTime);
fprintf('Maximum single Zhao detection time: %.3f s (trial %d of %d)\n', ...
        maxZhaoTime,maxTimeTrial,T);
zhaoPrintSnapshot('one',beforeOne,afterOne);
fprintf('One-run sampled peak working set:  %.2f MiB (%d samples)\n', ...
        peakOne,nOne);
fprintf('One-run peak above start:          %.2f MiB\n', ...
        benchmarkK8.SinglePeakIncrease_MiB);
zhaoPrintSnapshot('100',before100,after100);
fprintf('100-run sampled peak working set:  %.2f MiB (%d samples)\n', ...
        peak100,n100);
fprintf('100-run peak above start:          %.2f MiB\n', ...
        benchmarkK8.AllPeakIncrease_MiB);
fprintf('Note: sampled at ~100 ms on Windows; short peaks may be missed.\n');
fprintf('Memory is not summed across 100 runs.\n');
fprintf('========================================\n');

%% 6. Original detection accuracy metrics
fprintf('\n===== K=8 Zhao accuracy over 100 trials =====\n');
names = {'TP','FP','FN','Recall','Precision','F1','MinSpacing','SpacingRatio','ZhaoTime'};
for j = 1:size(stats,2)
    fprintf('%-15s = %.6f +/- %.6f\n', ...
        names{j},mean(stats(:,j)),std(stats(:,j)));
end
fprintf('Error              = N/A (Zhao does not estimate E)\n');
fprintf('Reference radius   = %.6e\n',supportRadius);
fprintf('FP >= 10 rate      = %.4f\n',mean(stats(:,2)>=10));

meanStats = mean(stats,1);
stdStats = std(stats,0,1);
summaryTable = table(N,K,T,meanStats(1),stdStats(1), ...
    meanStats(2),stdStats(2),meanStats(3),stdStats(3), ...
    meanStats(4),stdStats(4),meanStats(5),stdStats(5), ...
    meanStats(6),stdStats(6),meanStats(7),stdStats(7), ...
    meanStats(8),stdStats(8), ...
    'VariableNames',{'N','K','Trials', ...
    'TP_mean','TP_std','FP_mean','FP_std','FN_mean','FN_std', ...
    'Recall_mean','Recall_std','Precision_mean','Precision_std', ...
    'F1_mean','F1_std','MinSpacing_mean','MinSpacing_std', ...
    'SpacingRatio_mean','SpacingRatio_std'});

writetable(summaryTable,summaryCsv);
writetable(struct2table(benchmarkK8),benchCsv);
save(resultMat,'stats','metadata','summaryTable','benchmarkK8','-v7.3');
fprintf('\nSaved %s\nSaved %s\nSaved %s\n', ...
    resultMat,summaryCsv,benchCsv);



%% Original Zhao detector, unchanged
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


%% Memory helpers
function snapshot = zhaoMemorySnapshot()
snapshot = struct('matlabMiB',NaN,'workingSetMiB',NaN);
if ~ispc
    return;
end
try
    m = memory;
    snapshot.matlabMiB = double(m.MemUsedMATLAB)/(1024^2);
catch
    % Not every MATLAB installation exposes all memory details.
end
try
    proc = System.Diagnostics.Process.GetCurrentProcess();
    proc.Refresh();
    snapshot.workingSetMiB = double(proc.WorkingSet64)/(1024^2);
catch
end
end

function zhaoPrintSnapshot(label,before,after)
fprintf('%s MATLAB memory:         %.2f -> %.2f MiB (change %+.2f MiB)\n', ...
    label,before.matlabMiB,after.matlabMiB, ...
    after.matlabMiB-before.matlabMiB);
fprintf('%s process working set:   %.2f -> %.2f MiB (change %+.2f MiB)\n', ...
    label,before.workingSetMiB,after.workingSetMiB, ...
    after.workingSetMiB-before.workingSetMiB);
end

function extra = zhaoPeakIncrease(peak,baseline)
if isfinite(peak) && isfinite(baseline)
    extra = max(peak-baseline,0);
else
    extra = NaN;
end
end

function mon = zhaoStartMonitor()
% A separate PowerShell process samples the MATLAB process RAM ~every 100 ms.
% An in-MATLAB timer would not reliably fire during blocking SVD.
mon = struct('active',false,'folder','','phaseFile','','readyFile','', ...
             'resultFile','','pid',NaN);
if ~ispc
    warning('Peak memory monitoring requires Windows. Timing will still work.');
    return;
end

try
    proc = System.Diagnostics.Process.GetCurrentProcess();
    mon.pid = double(proc.Id);
    mon.folder = tempname(tempdir);
    mkdir(mon.folder);
    mon.phaseFile = fullfile(mon.folder,'phase.txt');
    mon.readyFile = fullfile(mon.folder,'ready.txt');
    mon.resultFile = fullfile(mon.folder,'result.txt');
    psFile = fullfile(mon.folder,'monitor.ps1');
    zhaoWriteText(mon.phaseFile,'SINGLE');

    psLines = {
        'param([int]$TargetPid,[string]$PhaseFile,[string]$ReadyFile,[string]$ResultFile)'
        '$ErrorActionPreference = "Stop"'
        '$culture = [System.Globalization.CultureInfo]::InvariantCulture'
        '$peakSingle = 0.0; $peakAll = 0.0; $nSingle = 0; $nAll = 0'
        '$lastPhase = "SINGLE"'
        '$deadline = (Get-Date).AddHours(8)'
        '[System.IO.File]::WriteAllText($ReadyFile,"ready")'
        'while ((Get-Date) -lt $deadline) {'
        '  try { $phase = ([System.IO.File]::ReadAllText($PhaseFile)).Trim() } catch { $phase = $lastPhase }'
        '  if ($phase -eq "STOP") { break }'
        '  try { $p = Get-Process -Id $TargetPid -ErrorAction Stop } catch { break }'
        '  $ws = [double]$p.WorkingSet64 / 1MB'
        '  if ($phase -eq "SINGLE") {'
        '    if ($ws -gt $peakSingle) { $peakSingle = $ws }'
        '    $nSingle++'
        '  }'
        '  if ($phase -eq "SINGLE" -or $phase -eq "GROUP") {'
        '    if ($ws -gt $peakAll) { $peakAll = $ws }'
        '    $nAll++'
        '  }'
        '  $lastPhase = $phase'
        '  Start-Sleep -Milliseconds 100'
        '}'
        '[System.IO.File]::WriteAllLines($ResultFile,[string[]]@($peakSingle.ToString($culture),$peakAll.ToString($culture),$nSingle.ToString(),$nAll.ToString()))'
        };

    fid = fopen(psFile,'w');
    if fid < 0, error('Cannot write the PowerShell monitoring script.'); end
    for j = 1:numel(psLines)
        fprintf(fid,'%s\n',psLines{j});
    end
    fclose(fid);

    processInfo = System.Diagnostics.ProcessStartInfo();
    processInfo.FileName = 'powershell.exe';
    processInfo.Arguments = sprintf([ ...
        '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass ' ...
        '-File "%s" -TargetPid %.0f -PhaseFile "%s" ' ...
        '-ReadyFile "%s" -ResultFile "%s"'], ...
        psFile,mon.pid,mon.phaseFile,mon.readyFile,mon.resultFile);
    processInfo.UseShellExecute = false;
    processInfo.CreateNoWindow = true;
    child = System.Diagnostics.Process.Start(processInfo);

    for trial = 1:60
        if isfile(mon.readyFile)
            mon.active = true;
            fprintf('Peak memory sampler started (Windows working set, ~100 ms).\n');
            return;
        end
        if child.HasExited, break; end
        pause(0.1);
    end
    warning('Cannot start PowerShell sampler; peak memory will be N/A.');
catch ME
    warning('Peak memory sampler unavailable: %s',ME.message);
end
% Keep partial folder for debugging if start fails; it is safe to delete.
end

function zhaoSetPhase(mon,phase)
if ~mon.active, return; end
try
    zhaoWriteText(mon.phaseFile,phase);
catch ME
    warning('Failed to update sampler state: %s',ME.message);
end
end

function [onePeak,allPeak,nOne,nAll] = zhaoReadMonitor(mon,timeoutSec)
onePeak = NaN;
allPeak = NaN;
nOne = 0;
nAll = 0;
if ~mon.active, return; end
clock = tic;
while ~isfile(mon.resultFile) && toc(clock) < timeoutSec
    pause(0.1);
end
if ~isfile(mon.resultFile)
    warning('Memory sampler did not produce results.');
    return;
end
try
    vals = sscanf(fileread(mon.resultFile),'%f');
    if numel(vals) == 4
        nOne = vals(3);
        nAll = vals(4);
        if nOne > 0, onePeak = vals(1); end
        if nAll > 0, allPeak = vals(2); end
    end
catch ME
    warning('Unable to read memory monitor results: %s',ME.message);
end
end

function zhaoWriteText(file,textValue)
fid = fopen(file,'w');
if fid < 0, error('Cannot open monitor state file.'); end
fprintf(fid,'%s',textValue);
fclose(fid);
end

function zhaoCleanupMonitor(mon)
if isempty(mon.folder), return; end
try
    if mon.active && ~isfile(mon.resultFile)
        zhaoSetPhase(mon,'STOP');
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
