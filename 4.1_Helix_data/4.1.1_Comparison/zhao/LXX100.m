addpath('./function');
N = 4000;

%% =========================================================
%  生成螺旋线数据（与原 luoxuanxian.m 一致）
% ==========================================================

t = linspace(-pi, pi, N);

x = sin(t);
y = cos(t);
z = t ;

r = [x; y; z];


%% 实验参数
K = floor(0.05 * N);           % 异常值数量：5%
k = 15;                       % MOD 的近邻数量
T = 100;                      % 随机重复实验次数

% 无异常的真实函数值：每次试验均从这里重新开始
f_true = frankgaowei(r)';
f_true = f_true(:);

% 固定随机种子，保证结果可复现
rng(1);

%% =========================================================
%  存储 100 次实验的各项指标
% ==========================================================
Recall_all       = zeros(T, 1);
Precision_all    = zeros(T, 1);
F1_all           = zeros(T, 1);
TP_all = zeros(T, 1);
FP_all = zeros(T, 1);
FN_all = zeros(T, 1);

%% =========================================================
%  重复随机实验 100 次
% ==========================================================
for trial = 1:T

    %% 每次重新初始化，避免异常值在不同试验间累积
    E = zeros(N, 1);
    f = f_true;

    %% 随机生成 K 个互不重复的异常点位置
    index = sort(randperm(N, K));

    %% 真实异常点之间的最小欧氏距离及其与参考半径之比
    if K >= 2
        outlier_points = r(:, index);
        min_spacing = inf;

        for j = 1:K-1
            displacement = outlier_points(:, j+1:K) - outlier_points(:, j);
            distances = sqrt(sum(displacement.^2, 1));
            min_spacing = min(min_spacing, min(distances));
        end

    else

    end

    %% 加入随机异常值：保持原程序 0.7~1.5 的正向扰动
    for j = 1:K
        noise = 0.05 + 0.05 * rand(1);
        E(index(j)) = noise;
    end
    f = f + E;

    %% MOD 异常检测与去噪
    X_noisy = [r; f'];
    [X_denoised, detected_outliers] = MOD(X_noisy, k);

    % MOD 应返回异常点下标；也兼容长度为 N 的逻辑掩码
    if islogical(detected_outliers)
        if numel(detected_outliers) ~= N
            error('MOD 返回的逻辑掩码长度必须等于 N。');
        end
        ep_indices = find(detected_outliers(:));
    else
        ep_indices = unique(detected_outliers(:));
    end

    if ~isnumeric(ep_indices) || ...
            any(~isfinite(ep_indices)) || ...
            any(ep_indices < 1 | ep_indices > N | ep_indices ~= floor(ep_indices))
        error('MOD 返回的异常点下标必须为 1 到 N 的整数。');
    end

    %% 恢复误差（采用与 LXX100.m 相同的定义）
    % 假设 X_denoised 与原始点逐列对应，并保留目标函数值。
    % 兼容 MOD 返回 4 x N、N x 4 或长度 N 的函数值向量。
    if isequal(size(X_denoised), size(X_noisy))
        f_approx = X_denoised(end, :)';
    elseif isequal(size(X_denoised), size(X_noisy'))
        f_approx = X_denoised(:, end);
    elseif isvector(X_denoised) && numel(X_denoised) == N
        f_approx = X_denoised(:);
    else
        error(['无法计算 Error：MOD 的 X_denoised 输出没有与原始 N 个点一一对应。' ...
               '请先核实 MOD 返回值的结构。']);
    end


    %% =====================================================
    %  Recall、Precision、F1
    % ======================================================
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

    if 2 * TP + FP + FN == 0
        F1 = 0;
    else
        F1 = 2 * TP / (2 * TP + FP + FN);
    end

    %% 保存本次实验结果
    Recall_all(trial) = Recall;
    Precision_all(trial) = Precision;
    F1_all(trial) = F1;

end

%% =========================================================
%  100 次实验的均值和标准差
% ==========================================================
Recall_mean = mean(Recall_all);
Recall_std  = std(Recall_all);

Precision_mean = mean(Precision_all);
Precision_std  = std(Precision_all);

F1_mean = mean(F1_all);
F1_std  = std(F1_all);

all_detected_count = sum(TP_all == K);



%% =========================================================
%  显示最终结果（与 LXX100.m 一致）
% ==========================================================
fprintf('\n==============================\n');
fprintf('N = %d, K = %d, Trials = %d\n', N, K, T);

fprintf('Recall    = %.6f ± %.6f\n', ...
    Recall_mean, Recall_std);
fprintf('Precision = %.6f ± %.6f\n', ...
    Precision_mean, Precision_std);
fprintf('F1        = %.6f ± %.6f\n', ...
    F1_mean, F1_std);
fprintf('All true outliers detected = %d/%d\n', all_detected_count, T);


fprintf('==============================\n');
