addpath('./function');

%% =========================================================
% 1. 生成螺旋线数据
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
z_ref = t_ref ;

M_ref = [x_ref; y_ref; z_ref];

h = fillh(r, M_ref);
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
[A, geometry] = MMLSforA(r,epsilon,d,m,h,delta);
Dproj = geometry.Dproj;
omega_delta = geometry.omega_delta;
U_mls = eye(N)-A;
UN_mls = vecnorm(U_mls,2,1)';
tA = toc;
%% 直接读取保存的矩阵 A
%% =========================================================
% 4. 实验设置
% ==========================================================
T = 100;

% 异常点位置和幅度使用原随机数流
rng(1);
noiseStream = RandStream('mt19937ar','Seed',2025);


Recall_all       = zeros(T,1);
Precision_all    = zeros(T,1);
F1_all           = zeros(T,1);
Error_all        = zeros(T,1);
Q0_all           = nan(T,1);
Q0Condition_all  = false(T,1);
FullDetection_all = false(T,1);

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
        noise = 0.05 + 0.05*rand(1);
        direction = 2*randi([0,1])-1;
        E(index(j)) = direction*noise;
    end

    %% 按文章定义独立生成 U(-0.02,0.02) 乘性噪声
    xi = 0.04*rand(noiseStream,N,1)-0.02;
    noise_small = xi.*f_true;

    %% 最终观测函数值
    f = f_true + E + noise_small;

    if K >= 2
        % 手稿中的最小投影分离距离 q_tilde_0。
        D_out = Dproj(index,index);
        D_out(1:K+1:end) = inf;
        Q0_all(trial) = min(D_out,[],'all');
        Q0Condition_all(trial) = support_radius + omega_delta <= Q0_all(trial);
    end

    %% MMLS 异常值检测
    [Ep,Index,f_approx] = MMLSforoutlier(A,f,U_mls,UN_mls);

    Ep = Ep(:);

    %% 恢复误差
    Error_all(trial) = norm(E-Ep,2)/K;

    %% Recall、Precision、F1
    ep_indices = find(abs(Ep)>1e-12);

    TP = numel(intersect(index,ep_indices));
    FP = numel(setdiff(ep_indices,index));
    FN = numel(setdiff(index,ep_indices));
    FullDetection_all(trial) = (TP == K && FN == 0 && FP == 0);

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

Q0_mean = mean(Q0_all,'omitnan');
Q0_std = std(Q0_all,'omitnan');
FullDetection_count = sum(FullDetection_all);

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
fprintf('omega_delta           = %.6e\n',omega_delta);
fprintf('q_tilde_0             = %.6e +/- %.6e\n',Q0_mean,Q0_std);
fprintf('delta + omega_delta   = %.6e\n',support_radius+omega_delta);
fprintf('Separation condition  = %d / %d trials\n', ...
    sum(Q0Condition_all),T);

fprintf('Full detection count  = %d / %d trials\n', ...
    FullDetection_count,T);

fprintf('A build time          = %.3f s\n',tA);
fprintf('==============================\n');
end
