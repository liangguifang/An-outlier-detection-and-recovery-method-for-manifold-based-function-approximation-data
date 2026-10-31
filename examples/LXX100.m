addpath('./function');

%% =========================================================
% 1. 生成螺旋线数据
% ==========================================================
N = 1000;

t = linspace(-pi,pi,N);
x = sin(t);
y = cos(t);
z = t;

x_norm = (x-min(x))/(max(x)-min(x));
y_norm = (y-min(y))/(max(y)-min(y));
z_norm = (z-min(z))/(max(z)-min(z));

r = [x_norm;y_norm;z_norm];

%% =========================================================
% 2. 计算填充距离
% ==========================================================
N_ref = 40000;

t_ref = linspace(-pi,pi,N_ref);
x_ref = sin(t_ref);
y_ref = cos(t_ref);
z_ref = t_ref;

x_ref_norm = (x_ref-min(x_ref))/(max(x_ref)-min(x_ref));
y_ref_norm = (y_ref-min(y_ref))/(max(y_ref)-min(y_ref));
z_ref_norm = (z_ref-min(z_ref))/(max(z_ref)-min(z_ref));

M_ref_norm = [x_ref_norm;y_ref_norm;z_ref_norm];

h = fillh(r,M_ref_norm);

K = floor(0.05*N);
support_radius = 20*h;
delta=support_radius;
%% =========================================================
% 3. 真实函数值及矩阵 A
% ==========================================================
f_true = frankgaowei(r)';
f_true = f_true(:);

epsilon = 1e-6;
d = 1;
m = 3;

tic;
A = MMLSforA(r,epsilon,d,m,h,delta);
tA = toc;
%% 直接读取保存的矩阵 A
%load('LXX100(4000).mat','A');

%% =========================================================
% 4. 实验设置
% ==========================================================
T = 5;

% 异常点位置和幅度使用原随机数流
rng(1);

% 小噪声使用独立随机数流
noiseStream = RandStream('mt19937ar','Seed',2026);

Recall_all       = zeros(T,1);
Precision_all    = zeros(T,1);
F1_all           = zeros(T,1);
Error_all        = zeros(T,1);
MinSpacing_all   = nan(T,1);
SpacingRatio_all = nan(T,1);

%% =========================================================
% 5. 重复实验100次
% ==========================================================
for trial = 1:T

    %% 每次从干净数据开始
    E = zeros(N,1);

    %% 生成随机异常位置
    index = sort(randperm(N,K));

    %% 添加随机正负异常值
    for j = 1:K
        noise = 0.5 + 0.5*rand(1);
        direction = 2*randi([0,1])-1;
        E(index(j)) = direction*noise;
    end

    %% 添加2%的随机正负乘性噪声
    w = 0.02*rand(noiseStream,N,1);

    direction_small =2*randi(noiseStream,[0,1],N,1)-1;

    noise_small = direction_small .* w .* f_true;

    %% 最终观测函数值
    f = f_true + E + noise_small;

    %% 计算真实异常点最小间距
    if K >= 2
        outlier_points = r(:,index);
        min_spacing = inf;

        for j = 1:K-1
            displacement = outlier_points(:,j+1:K) - outlier_points(:,j);

            distances = sqrt(sum(displacement.^2,1));

            min_spacing = min(min_spacing,min(distances));
        end

        MinSpacing_all(trial) = min_spacing;

        SpacingRatio_all(trial) = min_spacing/support_radius;
    end

    %% MMLS 异常值检测
    [Ep,Index,f_approx] = MMLSforoutlier(A,f);

    Ep = Ep(:);

    %% 恢复误差
    Error_all(trial) = norm(E-Ep,2)/K;

    %% Recall、Precision、F1
    ep_indices = find(abs(Ep)>1e-12);

    TP = numel(intersect(index,ep_indices));
    FP = numel(setdiff(ep_indices,index));
    FN = numel(setdiff(index,ep_indices));

    Recall_all(trial) = TP/(TP+FN);

    if TP+FP == 0
        Precision_all(trial) = 0;
    else
        Precision_all(trial) = TP/(TP+FP);
    end

    if 2*TP+FP+FN == 0
        F1_all(trial) = 0;
    else
        F1_all(trial) = ...
            2*TP/(2*TP+FP+FN);
    end
end

%% =========================================================
% 6. 统计结果
% ==========================================================
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

%% =========================================================
% 7. 输出结果
% ==========================================================
fprintf('\n==============================\n');
fprintf('N = %d, K = %d, Trials = %d\n',N,K,T);

fprintf('Noise = 2%% multiplicative noise\n');

fprintf('Recall    = %.6f +/- %.6f\n', ...
    Recall_mean,Recall_std);

fprintf('Precision = %.6f +/- %.6f\n', ...
    Precision_mean,Precision_std);

fprintf('F1        = %.6f +/- %.6f\n', ...
    F1_mean,F1_std);

fprintf('Error     = %.6e +/- %.6e\n', ...
    Error_mean,Error_std);

fprintf('Support radius        = %.6e\n',support_radius);

fprintf('Min outlier spacing   = %.6e +/- %.6e\n', ...
    MinSpacing_mean,MinSpacing_std);

fprintf('Min spacing / radius  = %.6f +/- %.6f\n', ...
    SpacingRatio_mean,SpacingRatio_std);

fprintf('A build time          = %.3f s\n',tA);
fprintf('==============================\n');