% Zhao Klein bottle detection benchmark.
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir,'function'));

%% 克莱因瓶数据，与 Our 方法共用训练样本
N_list = [1000,2000,4000];
geometrySeed = 1;
outlierMagnitude = [0.6,1.1];

for N = N_list
[r,u,v,f_true] = MMLSKleinBottleSamples(N,geometrySeed);

%% MOD 算法参数及重复实验设置
K = floor(0.05*N);           % 5% 的数据点为异常点
k = 15;                     % MOD 的近邻数量
T = 100;                    % 100 次重复随机实验

% 与 Our 方法使用相同的真实异常位置、幅值和符号。
outlierStream = RandStream('mt19937ar','Seed',1);

%% 保存每次实验的各项指标
Recall_all     = zeros(T,1);
Precision_all  = zeros(T,1);
F1_all         = zeros(T,1);

TP_all = zeros(T,1);
FP_all = zeros(T,1);
FN_all = zeros(T,1);

%% 重复随机实验 100 次
for trial = 1:T

    [index,E] = MMLSSignedOutliers(outlierStream,N,K,outlierMagnitude);

    %% 真实异常值点之间的最小四维欧氏距离
    % 此指标只依赖点的坐标，不需要 h 或支撑半径
    if K >= 2
        outlier_points = r(:,index);
        min_spacing = inf;

        for j = 1:K-1
            displacement = outlier_points(:,j+1:K) - outlier_points(:,j);
            distances = sqrt(sum(displacement.^2,1));
            min_spacing = min(min_spacing,min(distances));
        end

    end

    %% 生成含异常值的函数数据
    f = f_true + E;

    %% MOD 异常值识别（保留 Kle.m 使用的算法）
    X_noisy = [r;f'];          % 5 x N
    [~,detected_outliers] = MOD(X_noisy,k);

    % MOD 可以返回异常点索引，或 N 元逻辑掩码
    if islogical(detected_outliers)
        if numel(detected_outliers) ~= N
            error('MOD 返回的逻辑掩码长度必须等于 N。');
        end
        ep_indices = find(detected_outliers(:));
    elseif isnumeric(detected_outliers)
        ep_indices = unique(detected_outliers(:));
    else
        error('MOD 返回的异常点位置必须是数值索引或逻辑掩码。');
    end

    if any(~isfinite(ep_indices)) || ...
       any(ep_indices < 1 | ep_indices > N | ep_indices ~= floor(ep_indices))
        error('MOD 返回的异常点索引必须是 1 到 N 的整数。');
    end

    %% TP、FP、FN、Recall、Precision、F1
    TP = numel(intersect(index,ep_indices));
    FP = numel(setdiff(ep_indices,index));
    FN = numel(setdiff(index,ep_indices));

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

    Recall_all(trial)    = Recall;
    Precision_all(trial) = Precision;
    F1_all(trial)        = F1;

end

%% 100 次实验的均值和标准差
Recall_mean = mean(Recall_all);
Recall_std  = std(Recall_all);

Precision_mean = mean(Precision_all);
Precision_std  = std(Precision_all);

F1_mean = mean(F1_all);
F1_std  = std(F1_all);

all_detected_count = sum(TP_all == K & FP_all == 0 & FN_all == 0);

%% 显示最终结果（参照 KB100.m，移除 Error 和 h 相关指标）
fprintf('\n==============================\n');
fprintf('Algorithm = MOD\n');
fprintf('N = %d, K = %d, Trials = %d\n',N,K,T);
fprintf('Recall    = %.6f ± %.6f\n',Recall_mean,Recall_std);
fprintf('Precision = %.6f ± %.6f\n',Precision_mean,Precision_std);
fprintf('F1        = %.6f ± %.6f\n',F1_mean,F1_std);
fprintf('Exact support recovered = %d/%d\n',all_detected_count,T);
fprintf('==============================\n');
end
