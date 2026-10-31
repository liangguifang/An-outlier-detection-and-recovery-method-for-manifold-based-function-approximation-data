addpath('./function')

%% =========================================================
%  克莱因瓶数据
% ==========================================================

% 1000 个点
Nu = 25;
Nv = 40;

% % 2000 个点
% Nu = 40;
% Nv = 50;

% % 4000 个点
% Nu = 50;
% Nv = 80;

u0 = (0:Nu-1) * (2*pi/Nu);
v0 = (0:Nv-1) * (2*pi/Nv);

[u,v] = meshgrid(u0,v0);

u = u(:)';
v = v(:)';

% 总数据点数
N = length(u);       % Nu*Nv = 1000

%% 克莱因瓶的参数方程
x1 = (2.*cos(v)+1).*cos(u);
x2 = (2.*cos(v)+1).*sin(u);
x3 = 2.*sin(v).*cos(u./2);
x4 = 2.*sin(v).*sin(u./2);

r = [x1;x2;x3;x4];

%% 填充距离
Nu_ref = 200;
Nv_ref = 200;

u_ref = linspace(0,2*pi,Nu_ref);
v_ref = linspace(0,2*pi,Nv_ref);

[uu,vv] = meshgrid(u_ref,v_ref);

uu = uu(:)';
vv = vv(:)';

x1_ref = (2.*cos(vv)+1).*cos(uu);
x2_ref = (2.*cos(vv)+1).*sin(uu);
x3_ref = 2.*sin(vv).*cos(uu./2);
x4_ref = 2.*sin(vv).*sin(uu./2);

M_ref = [x1_ref;x2_ref;x3_ref;x4_ref];

h = fillh(r,M_ref);

K = floor(0.05*N);    % 异常值数量（5%）

%% 支撑半径设置

support_radius = 4*h;
delta=support_radius ;
%% 无异常的真实函数值
f_true = ( ...
    1/2 .* sin(4 .* u) ...
    + 1/5 .* cos(2 .* v).^2 ...
    + 1/6 .* cos(4 .* u).*sin(2 .* v).^2 ...
    )';

epsilon = 1e-6;
d = 2;
m = 3;

%% 计算 A：只计算一次
tic;
A = MMLSforA(r,epsilon,d,m,h,delta);
tA = toc;
%load('KB4000A.mat','A');


%% =========================================================
%  重复随机实验 100 次
% ==========================================================
T = 100;

% 为了论文结果能够重复
rng(1);
noiseStream = RandStream('mt19937ar','Seed',2026);

Recall_all    = zeros(T,1);
Precision_all = zeros(T,1);
F1_all        = zeros(T,1);
Error_all     = zeros(T,1);
MinSpacing_all  = zeros(T,1);
SpacingRatio_all = zeros(T,1);


TP_all = zeros(T,1);
FP_all = zeros(T,1);
FN_all = zeros(T,1);


for trial = 1:T

    %% 每次实验必须重新初始化
    E = zeros(N,1);

    % 每次都从干净函数值重新开始
    %f = f_true;

    %% 生成随机异常值索引
    index = randperm(N,K);
    index = sort(index);

    %% 真实异常值点之间的最小欧氏距离，以及它与支撑半径的比值
    % r 的列是数据点；在螺旋线上为三维坐标，在克莱因瓶上为四维坐标。
    outlier_points = r(:,index);
    min_spacing = inf;
    for j = 1:K-1
        displacement = outlier_points(:,j+1:K) - outlier_points(:,j);
        distances = sqrt(sum(displacement.^2,1));
        min_spacing = min(min_spacing, min(distances));
    end
    MinSpacing_all(trial) = min_spacing;
    SpacingRatio_all(trial) = min_spacing / support_radius;


    %% 添加随机符号和随机大小的异常值
    for j = 1:K
        noise = 0.6 + 0.5*rand(1);       % 随机幅度：0.6 到 1.0
        direction = (randi([0,1])*2)-1;  % 随机符号：+1 或 -1
        E(index(j)) = direction*noise;
    end
    % w = 0.02*rand(noiseStream,N,1);
    % 
    % direction_small =2*randi(noiseStream,[0,1],N,1)-1;
    % 
    % noise_small = direction_small .* w .* f_true;

    %% 最终观测函数值
    f = f_true + E ;%+ noise_small;

    %% 异常检测
    [Ep,Index,f_approx] = MMLSforoutlier(A,f);

    %% 恢复误差
    erros1 = abs(f-E-f_approx);
    erros = norm(E-Ep,2)/K;
    Error_all(trial) = erros;


    %% =====================================================
    %  Recall、Precision、F1
    % ======================================================

    % 检测到的异常位置
    % 用阈值比 Ep ~= 0 更稳妥
    ep_indices = find(abs(Ep) > 1e-12);

    % TP：正确检测到的异常点
    TP = numel(intersect(index,ep_indices));

    % FP：正常点被误认为异常
    FP = numel(setdiff(ep_indices,index));

    % FN：真正异常但没有检测出来
    FN = numel(setdiff(index,ep_indices));

    TP_all(trial) = TP;
    FP_all(trial) = FP;
    FN_all(trial) = FN;

    %% Recall
    if TP+FN == 0
        Recall = 0;
    else
        Recall = TP/(TP+FN);
    end

    %% Precision
    if TP+FP == 0
        Precision = 0;
    else
        Precision = TP/(TP+FP);
    end

    %% F1
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


%% =========================================================
%  100 次实验的均值和标准差
% ==========================================================

Recall_mean = mean(Recall_all);
Recall_std  = std(Recall_all);

Precision_mean = mean(Precision_all);
Precision_std  = std(Precision_all);

F1_mean = mean(F1_all);
F1_std  = std(F1_all);

Error_mean = mean(Error_all);
Error_std  = std(Error_all);

MinSpacing_mean = mean(MinSpacing_all);
MinSpacing_std  = std(MinSpacing_all);

SpacingRatio_mean = mean(SpacingRatio_all);
SpacingRatio_std  = std(SpacingRatio_all);



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

fprintf('Support radius        = %.6e\n',support_radius);
fprintf('Min outlier spacing   = %.6e ± %.6e\n', ...
    MinSpacing_mean,MinSpacing_std);
fprintf('Min spacing / radius  = %.6f ± %.6f\n', ...
    SpacingRatio_mean,SpacingRatio_std);

fprintf('A time    = %.6e s\n',tA);
fprintf('==============================\n');