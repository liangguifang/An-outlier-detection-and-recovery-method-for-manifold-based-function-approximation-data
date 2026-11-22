% Helix detection and recovery with multiplicative noise.
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir,'function'));

%% 1. 生成螺旋线数据
% 每次运行初始化独立的小噪声随机流，供所有样本规模使用。
noiseStream = RandStream('mt19937ar','Seed','shuffle');
N_list = [1000,2000,4000];
for N = N_list

t = linspace(-pi, pi, N);

x = sin(t);
y = cos(t);
z = t ;
r = [x; y; z];

K = floor(0.05*N);
outlierMagnitude = [0.05,0.1];

%% 3. 真实函数值及矩阵 A
f_true = frankgaowei(r)';
f_true = f_true(:);

d = 1;
m = 3;

buildTimer = tic;
[A, neighborCount, localRadii, geometry] = buildLocalPcaMLS(r.',d,m,[],true,false);
tA = toc(buildTimer);
% 用最大局部半径作为分离条件中的全局半径上界。
support_radius = max(localRadii);
omega_delta = geometry.omega_delta;
U_mls = speye(N)-A;
UN_mls = vecnorm(U_mls,2,1)';

%% 4. 实验设置
T = 100;
noiseLevel = 0.02;

% 重置异常值种子，使每轮与无噪实验配对。
outlierStream = RandStream('mt19937ar','Seed',1);

Recall_all       = zeros(T,1);
Precision_all    = zeros(T,1);
F1_all           = zeros(T,1);
Error_all        = zeros(T,1);
Q0_all           = nan(T,1);
outlierIndices   = zeros(K,T);
outlierValues    = zeros(K,T);
FullDetection_all = false(T,1);

%% 5. 重复实验100次
for trial = 1:T

    [index,E] = MMLSSignedOutliers(outlierStream,N,K,outlierMagnitude);
    outlierIndices(:,trial) = index(:);
    outlierValues(:,trial) = E(index);

    %% 先注入异常值，再对每个观测独立施加 U(-0.02,0.02) 乘性小噪声。
    f_corrupted = f_true + E;
    xi = (2*noiseLevel)*rand(noiseStream,N,1)-noiseLevel;
    f = (1+xi).*f_corrupted;

    %% MMLS 异常值检测，使用基于观测中位数的默认停止条件。
    Ep = MMLSforoutlier(A,f,U_mls,UN_mls);

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

%% 6. 统计结果
Recall_mean = mean(Recall_all);
Recall_std = std(Recall_all);

Precision_mean = mean(Precision_all);
Precision_std = std(Precision_all);

F1_mean = mean(F1_all);
F1_std = std(F1_all);

Error_mean = mean(Error_all);
Error_std = std(Error_all);

% 使用各轮真实异常位置，在识别结束后计算分离距离。
for trial = 1:T
    Q0_all(trial) = MMLSProjectedSeparation(geometry,outlierIndices(:,trial));
end
Q0Condition_all = support_radius + omega_delta <= Q0_all;
Q0_mean = mean(Q0_all,'omitnan');
Q0_std = std(Q0_all,'omitnan');
FullDetection_count = sum(FullDetection_all);

%% 7. 输出结果
fprintf('\n==============================\n');
fprintf('N = %d, K = %d, Trials = %d\n',N,K,T);

fprintf('Noise = %.0f%% multiplicative noise applied after outliers\n',100*noiseLevel);

fprintf('Recall    = %.6f +/- %.6f\n', ...
    Recall_mean,Recall_std);

fprintf('Precision = %.6f +/- %.6f\n', ...
    Precision_mean,Precision_std);

fprintf('F1        = %.6f +/- %.6f\n', ...
    F1_mean,F1_std);

fprintf('Error     = %.6e +/- %.6e\n', ...
    Error_mean,Error_std);

fprintf('Neighbors per row     = %d to %d\n',min(neighborCount),max(neighborCount));
fprintf('Search radius sigma   = %.6e\n',geometry.sigma);
fprintf('Local support radii   = [%.6e, %.6e]\n',min(localRadii),support_radius);
fprintf('omega_delta           = %.6e\n',omega_delta);
fprintf('q_tilde_0             = %.6e +/- %.6e\n',Q0_mean,Q0_std);
fprintf('max(delta_j) + omega_delta = %.6e\n',support_radius+omega_delta);
fprintf('Separation condition  = %d / %d trials\n', ...
    sum(Q0Condition_all),T);

fprintf('Full detection count  = %d / %d trials\n', ...
    FullDetection_count,T);

fprintf('A build time          = %.3f s\n',tA);
fprintf('==============================\n');
end
