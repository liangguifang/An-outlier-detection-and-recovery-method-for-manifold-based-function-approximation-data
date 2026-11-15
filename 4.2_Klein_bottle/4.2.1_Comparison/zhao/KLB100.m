
addpath('./function');

%% =========================================================
%  克莱因瓶数据：与 KB100.m 使用相同的二维参数网格
% ==========================================================
N=4000;
u = linspace(0, 2*pi, N);
v = linspace(0, 2*pi, N);

%% 克莱因瓶的参数方程
x1 = (2.*cos(v)+1).*cos(u);
x2 = (2.*cos(v)+1).*sin(u);
x3 = 2.*sin(v).*cos(u./2);
x4 = 2.*sin(v).*sin(u./2);

r = [x1;x2;x3;x4];

%% 无异常的真实函数值（与 KB100.m 完全一致）
f_true = ( 1/2 .* sin(4 .* u) + 1/5 .* cos(2 .* v).^2 + 1/6 .* cos(4 .* u).*sin(2 .* v).^2 )';                      % N x 1

%% =========================================================
%  MOD 算法参数及重复实验设置
% ==========================================================
K = floor(0.05*N);           % 5% 的数据点为异常点
k = 15;                     % MOD 的近邻数量
T = 100;                    % 100 次重复随机实验

% 与 KB100.m 相同的随机种子
rng(1);

%% 保存每次实验的各项指标
Recall_all     = zeros(T,1);
Precision_all  = zeros(T,1);
F1_all         = zeros(T,1);


TP_all = zeros(T,1);
FP_all = zeros(T,1);
FN_all = zeros(T,1);

%% =========================================================
%  重复随机实验 100 次
% ==========================================================
for trial = 1:T

    %% 每次实验都从干净函数值重新初始化
    E = zeros(N,1);
    f = f_true;

    %% 生成随机异常值索引（不重复）
    index = randperm(N,K);
    index = sort(index);

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

    %% 添加与 KB100.m 一致的随机异常值
    % 随机幅度属于 [0.6,1.1)，随机符号为 +1 或 -1
    for j = 1:K
        noise = 0.6 + 0.5*rand(1);
        direction = (randi([0,1])*2)-1;
        E(index(j)) = direction*noise;
    end

    %% 生成含异常值的函数数据
    f = f + E;

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

    %% =====================================================
    %  TP、FP、FN、Recall、Precision、F1
    % ======================================================
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
%  显示最终结果（参照 KB100.m，移除 Error 和 h 相关指标）
% ==========================================================
fprintf('\n==============================\n');
fprintf('Algorithm = MOD\n');
fprintf('N = %d, K = %d, Trials = %d\n',N,K,T);
fprintf('Recall    = %.6f ± %.6f\n',Recall_mean,Recall_std);
fprintf('Precision = %.6f ± %.6f\n',Precision_mean,Precision_std);
fprintf('F1        = %.6f ± %.6f\n',F1_mean,F1_std);
fprintf('All true outliers detected = %d/%d\n',all_detected_count,T);
fprintf('==============================\n');
