% Zheng Klein bottle detection and recovery benchmark.
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir,'function'));

%% 克莱因瓶数据，与 Our 方法共用训练样本
N_list = [1000,2000,4000];
geometrySeed = 1;
outlierMagnitude = [0.6,1.1];

for N = N_list
[r,~,~,f_true] = MMLSKleinBottleSamples(N,geometrySeed);

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

fillTimer = tic;
h = fillh(r,M_ref);
fillTime = toc(fillTimer);

K = floor(0.05*N);    % 异常值数量（5%）

m = 3;
%% 计算 A：只计算一次
buildTimer = tic;
[A,support_radius] = L0ODRforA(r.',m,h,true);
tA = toc(buildTimer);

%% 重复随机实验 100 次
T = 100;

% 与 Our 方法使用相同的真实异常位置、幅值和符号。
outlierStream = RandStream('mt19937ar','Seed',1);

Recall_all    = zeros(T,1);
Precision_all = zeros(T,1);
F1_all        = zeros(T,1);
Error_all     = zeros(T,1);
MinSpacing_all  = zeros(T,1);
SpacingRatio_all = zeros(T,1);

TP_all = zeros(T,1);
FP_all = zeros(T,1);
FN_all = zeros(T,1);
AllDetectedCount = 0;
P = L0ODRprepare(A);
DetectionTime_all = zeros(T,1);

for trial = 1:T

    [index,E] = MMLSSignedOutliers(outlierStream,N,K,outlierMagnitude);

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

    %% 生成含异常值的函数数据
    f = f_true + E;

    %% 异常检测
    detectionTimer = tic;
    Ep = L0ODRforoutlier_fast(P,f,0.05);
    DetectionTime_all(trial) = toc(detectionTimer);

    %% 恢复误差
    Error_all(trial) = norm(E-Ep,2)/K;

    %% Recall、Precision、F1

    ep_indices = find(abs(Ep) > 1e-12);

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

    %% 保存当前这一次实验结果
    Recall_all(trial) = Recall;
    Precision_all(trial) = Precision;
    F1_all(trial) = F1;

end

%% 100 次实验的均值和标准差

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

fprintf('Fill distance time    = %.6e s\n',fillTime);
fprintf('A time                = %.6e s\n',tA);
fprintf('Detection total time  = %.6e s\n',sum(DetectionTime_all));
fprintf('All outliers detected = %d / %d trials\n',AllDetectedCount,T);
fprintf('==============================\n');
end
