% Klein bottle detection and recovery (outliers only).
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir,'function'));

N_list = [1000,2000,4000];
geometrySeed = 1;
d = 2;
m = 3;
ompTau = 0.5;  % 相邻残差降幅比的停止阈值
outlierMagnitude = [0.6,1.1];  % 克莱因瓶异常幅值。
for N = N_list
fprintf('\nN=%d, m=%d: generating training samples...\n',N,m);

[r,~,~,f_true] = MMLSKleinBottleSamples(N,geometrySeed);
K = floor(0.05*N);    % 异常值数量（5%）

%% 以原始查询点固定近邻，用逐行半径做 FLCS 和三次局部多项式拟合
fprintf('N=%d: building A with query-centered neighbors and local radii...\n',N);
buildTimer = tic;
[A,neighborCount,localRadii,geometry] = buildLocalPcaMLS(r.',d,m,[],true,false);
tA = toc(buildTimer);
support_radius = max(localRadii);
omega_delta = geometry.omega_delta;
fprintf(['N=%d: A completed in %.3f s; neighbors per row=%d to %d; ' ...
    'nonzero entries=%d (%.2f%%).\n'], ...
    N,tA,min(neighborCount),max(neighborCount),nnz(A),100*nnz(A)/numel(A));

% 跳过零幅值的异常识别试验。
if ~any(outlierMagnitude)
    continue;
end

% A、U 和列范数在同一 N 的 100 次识别中保持不变，只准备一次。
prepareTimer = tic;
U_mls = speye(N)-A;
UN_mls = sqrt(sum(U_mls.^2,1))';
tPrepare = toc(prepareTimer);

%% 重复随机实验 100 次
T = 100;

% 每个样本规模从相同的异常值随机流开始，与有噪声版逐轮配对。
outlierStream = RandStream('mt19937ar','Seed',1);
Recall_all    = zeros(T,1);
Precision_all = zeros(T,1);
F1_all        = zeros(T,1);
Error_all     = zeros(T,1);
FullDetection_all = false(T,1);

TP_all = zeros(T,1);
FP_all = zeros(T,1);
FN_all = zeros(T,1);
DetectionTime_all = zeros(T,1);
outlierIndices = zeros(K,T);

fprintf('N=%d: %d detection trials, K=%d, signed outlier magnitudes in [%g,%g), tau=%g.\n', ...
    N,T,K,outlierMagnitude(1),outlierMagnitude(2),ompTau);
for trial = 1:T

    [index,E] = MMLSSignedOutliers(outlierStream,N,K,outlierMagnitude);
    outlierIndices(:,trial) = index(:);

    f = f_true + E;

    %% 异常检测
    detectionTimer = tic;
    Ep = MMLSforoutlier(A,f,U_mls,UN_mls,ompTau);
    DetectionTime_all(trial) = toc(detectionTimer);
    if trial == 1 || mod(trial,10) == 0
        fprintf('N=%d: trial %d/%d, detection %.3f s, cumulative detection %.3f s.\n', ...
            N,trial,T,DetectionTime_all(trial),sum(DetectionTime_all(1:trial)));
    end

    %% 恢复误差
    Error_all(trial) = norm(E-Ep(:),2)/K;

    %% Recall、Precision、F1

    ep_indices = find(Ep(:) ~= 0);

    TP = numel(intersect(index,ep_indices));

    FP = numel(setdiff(ep_indices,index));

    FN = numel(setdiff(index,ep_indices));
    FullDetection_all(trial) = (TP == K && FP == 0 && FN == 0);

    TP_all(trial) = TP;
    FP_all(trial) = FP;
    FN_all(trial) = FN;

    if TP+FN == 0
        Recall = 0;
    else
        Recall = TP/(TP+FN);
    end

    if TP+FP == 0
        Precision = 0;
    else
        Precision = TP/(TP+FP);
    end

    if 2*TP+FP+FN == 0
        F1 = 0;
    else
        F1 = 2*TP/(2*TP+FP+FN);
    end

    %% 保存当前这一次实验结果
    Recall_all(trial) = Recall;
    Precision_all(trial) = Precision;
    F1_all(trial) = F1;

end

%% 100 次实验的均值和标准差

% 在识别完成后，使用原实验的真实异常位置计算投影分离距离。
Q0_all = nan(T,1);
for trial = 1:T
    Q0_all(trial) = MMLSProjectedSeparation(geometry,outlierIndices(:,trial));
end
Q0_mean = mean(Q0_all,'omitnan');
Q0_std = std(Q0_all,'omitnan');
Q0Condition_all = support_radius + omega_delta <= Q0_all;
Q0Ratio_all = Q0_all/(support_radius+omega_delta);
Q0Ratio_mean = mean(Q0Ratio_all,'omitnan');
Q0Ratio_std = std(Q0Ratio_all,'omitnan');

Recall_mean = mean(Recall_all);
Recall_std  = std(Recall_all);

Precision_mean = mean(Precision_all);
Precision_std  = std(Precision_all);

F1_mean = mean(F1_all);
F1_std  = std(F1_all);

Error_mean = mean(Error_all);
Error_std  = std(Error_all);

FullDetection_count = sum(FullDetection_all);

%% 显示最终结果
fprintf('\n==============================\n');
fprintf('N = %d, K = %d, Trials = %d\n',N,K,T);

fprintf('Recall    = %.6f ± %.6f\n', ...
    Recall_mean,Recall_std);

fprintf('Precision = %.6f ± %.6f\n', ...
    Precision_mean,Precision_std);

fprintf('F1        = %.6f ± %.6f\n', ...
    F1_mean,F1_std);

fprintf('Error     = %.6e ± %.6e\n', ...
    Error_mean,Error_std);

fprintf('Full detection count  = %d / %d trials\n', ...
    FullDetection_count,T);

fprintf('q_tilde_0             = %.6e +/- %.6e\n',Q0_mean,Q0_std);
fprintf('q_tilde_0 / (max(delta_j)+omega_delta) = %.6e +/- %.6e\n',Q0Ratio_mean,Q0Ratio_std);
fprintf('max(delta_j) + omega_delta = %.6e\n',support_radius+omega_delta);
fprintf('Separation condition  = %d / %d trials\n',sum(Q0Condition_all),T);
fprintf('A time    = %.6e s\n',tA);
fprintf('Detection setup time  = %.3f s\n',tPrepare);
fprintf('Detection total time  = %.3f s\n',sum(DetectionTime_all));
fprintf('Detection mean/max    = %.3f / %.3f s\n', ...
    mean(DetectionTime_all),max(DetectionTime_all));
fprintf('==============================\n');

end
