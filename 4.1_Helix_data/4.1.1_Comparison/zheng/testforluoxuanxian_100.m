addpath('./function');

%% =========================================================
%  1. 生成螺旋线数据（保留原 testforluoxuanxian.m 的设置）
% ==========================================================
N_list = [1000,2000,4000];
for N = N_list
t = linspace(-pi, pi, N);

x = sin(t);
y = cos(t);
z = t ;

r = [x; y; z];

%% =========================================================
% 2. 计算填充距离
% ==========================================================
N_ref = 40000;

t_ref = linspace(-pi, pi, N_ref);

x_ref = sin(t_ref);
y_ref = cos(t_ref);
z_ref = t_ref;

M_ref = [x_ref; y_ref; z_ref];

h = fillh(r, M_ref);


Xn = r';                   % N x 3
f_true = frankgaowei(r)';
f_true = f_true(:);                % N x 1
if numel(f_true) ~= N
    error('frankgaowei 返回的函数值数目与 N 不一致。');
end

%% =========================================================
%  2. 先计算矩阵 A（仅一次，不放进循环中）
% ==========================================================
dp = 3;                          % 原 L0ODR 默认多项式次数
Eps = h^(dp-1);              % 原 L0ODR 默认 OMP 停止参数
tic;
A = L0ODRforA(Xn,dp,h);
tA = toc;
%load('LXXA4000.mat');
%% =========================================================
%  3. 实验参数及结果存储
% ==========================================================
T = 100;
K = floor(0.05*N);               % 5% 随机异常点
rng(1);                          % 固定随机种子

% 原 LW 函数中的有效截断距离为 4*delta；
% 此处用未归一化的 Xn 计算点间距，保持相同几何尺度。
% 注意：原权重以坐标原点而非每个查询点为中心。
support_radius = 20*h;

Recall_all       = zeros(T,1);
Precision_all    = zeros(T,1);
F1_all           = zeros(T,1);
Error_all        = zeros(T,1);
ErrorTrue_all    = zeros(T,1);   % 保留原程序 erros2 指标
MinSpacing_all   = nan(T,1);
SpacingRatio_all = nan(T,1);

TP_all = zeros(T,1);
FP_all = zeros(T,1);
FN_all = zeros(T,1);
AllDetectedCount = 0;

%% =========================================================
%  4. 异常值识别循环 100 次
% ==========================================================
for trial = 1:T

    % 重新初始化，不累积上一次异常与噪声
    E = zeros(N,1);
    f = f_true;

    % 随机生成真实异常点下标
    index = sort(randperm(N,K));

    % 真实异常点之间的最小欧氏距离
    if K >= 2
        outlier_points = Xn(index,:).';
        min_spacing = inf;
        for j = 1:K-1
            displacement = outlier_points(:,j+1:K)-outlier_points(:,j);
            distances = sqrt(sum(displacement.^2,1));
            min_spacing = min(min_spacing,min(distances));
        end
        MinSpacing_all(trial) = min_spacing;
        SpacingRatio_all(trial) = min_spacing/support_radius;
    end

    % 保留原脚本中的小噪声：U[0,0.044)
    % noise1 = 0.044*rand(size(f));
    % f = f+noise1;

     for j = 1:K
        noise = 0.05 + 0.05*rand(1);
        direction = (randi([0,1])*2)-1;
        E(index(j)) = direction*noise;
    end
    %% 加入异常
    f = f + E;

    %% 只进行异常值识别，不再计算 A
    [Ep,Index,f_approx] = L0ODRforoutlier(A,f,0.05); 

    %% 恢复误差（与 LXX100.m 相同定义）
    Error_all(trial) = norm(E-Ep,2)/K;
    ErrorTrue_all(trial) = norm(E(index)-Ep(index),2)/K;

    %% 异常检测指标
    ep_indices = find(abs(Ep)>1e-12);
    TP = numel(intersect(index,ep_indices));
    FP = numel(setdiff(ep_indices,index));
    FN = numel(setdiff(index,ep_indices));
    if TP == K && FP == 0 && FN == 0
        AllDetectedCount = AllDetectedCount + 1;
    end

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

    Recall_all(trial) = Recall;
    Precision_all(trial) = Precision;
    F1_all(trial) = F1;
end

%% =========================================================
%  5. 100 次实验的均值和标准差
% ==========================================================
Recall_mean = mean(Recall_all);
Recall_std = std(Recall_all);

Precision_mean = mean(Precision_all);
Precision_std = std(Precision_all);

F1_mean = mean(F1_all);
F1_std = std(F1_all);

Error_mean = mean(Error_all);
Error_std = std(Error_all);

ErrorTrue_mean = mean(ErrorTrue_all);
ErrorTrue_std = std(ErrorTrue_all);

MinSpacing_mean = mean(MinSpacing_all);
MinSpacing_std = std(MinSpacing_all);

SpacingRatio_mean = mean(SpacingRatio_all);
SpacingRatio_std = std(SpacingRatio_all);

%% =========================================================
%  6. 输出（与 LXX100.m 的 100 次循环统计形式一致）
% ==========================================================
fprintf('\n==============================\n');
fprintf('N = %d, K = %d, Trials = %d\n',N,K,T);
fprintf('Recall    = %.6f ± %.6f\n',Recall_mean,Recall_std);
fprintf('Precision = %.6f ± %.6f\n',Precision_mean,Precision_std);
fprintf('F1        = %.6f ± %.6f\n',F1_mean,F1_std);
fprintf('Error     = %.6e ± %.6e\n',Error_mean,Error_std);
fprintf('Error (true outliers only) = %.6e ± %.6e\n', ...
    ErrorTrue_mean,ErrorTrue_std);

fprintf('Support radius        = %.6e\n',support_radius);
fprintf('Min outlier spacing   = %.6e ± %.6e\n', ...
    MinSpacing_mean,MinSpacing_std);
fprintf('Min spacing / radius  = %.6f ± %.6f\n', ...
    SpacingRatio_mean,SpacingRatio_std);
fprintf('A build time          = %.3f s\n',tA);
fprintf('All outliers detected = %d / %d trials\n',AllDetectedCount,T);
fprintf('==============================\n');
end
