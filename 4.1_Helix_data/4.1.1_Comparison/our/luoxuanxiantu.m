addpath('./function')
N=100;
%% 生成螺旋线数据
t = linspace(-pi, pi, N);
x = sin(t);
y = cos(t);
z = t;
r = [x; y; z];
%%


%填充距离
N_ref = 40000;   % 稠密参考点

t_ref = linspace(-pi, pi, N_ref);
x_ref = sin(t_ref);
y_ref = cos(t_ref);
z_ref = t_ref;

M_ref = [x_ref; y_ref; z_ref];

h=fillh(r,M_ref);


K = 5;%floor(0.05*N);    % 异常值数量（5%的数据点）

E = zeros(N, 1);      % 初始化偏差向量

f = frankgaowei(r);
f = f(:);


%% 生成随机异常值索引（不重复）
index = randperm(N, K);  % 从1到N中随机选取K个不重复的索引
index = sort(index);
%% 添加小噪声
%noise1 =0.02 * rand(size(f));  % 生成与 f 相同大小的均匀噪声矩阵，噪声值在 [0, 0.5] 之间
%f = f.*(1+ noise1);          % 将噪声添加到 f
%% 添加随机符号和大小的异常值
for j = 1:K
   noise =0.05+0.05*rand(1);   
   direction = (randi([0, 1]) * 2) - 1;  % 生成随机符号：结果为 1 或 -1
   E(index(j)) = direction * noise;      % 应用带符号的异常值
end
f = f + E;  % 生成含异常值的数据（f为观测值向量）
epsilon = 1e-6;  % 收敛阈值
d=1;m=3;
tic;

% 第一部分：计算投影和MLS矩阵A
delta = 20*h;
[A, geometry] = MMLSforA(r,epsilon,d,m,h,delta);

U_mls = speye(N)-sparse(A);
UN_mls = sqrt(sum(U_mls.^2,1))';

% 第二部分：异常值识别
[Ep,Index,f_approx] = MMLSforoutlier(A,f,U_mls,UN_mls);

t1=toc;
%% 计算检测准确率
ep_indices = find(Ep ~= 0);  % 检测到的异常点索引
TP = numel(intersect(index, ep_indices));  % 正确检测数
accuracy = TP / K * 100;      % 准确率
FP = numel(setdiff(ep_indices, index));    % 误检数
FN=numel(setdiff(index, ep_indices)); %漏检数

% --- 修正误差计算逻辑以便绘图 ---
erros1 = abs(f-E-f_approx);
scalar_error = norm(E-Ep,2)/K; % 这是一个标量，用于显示总误差
error_vec = abs(E - Ep);       % 这是一个向量，用于绘制散点图
% -----------------------------

%% 计算检测准确率
ep_indices = find(Ep ~= 0);  % 检测到的异常点索引
TP = numel(intersect(index, ep_indices));  % 正确检测数
accuracy = TP / K * 100;      % 准确率
FP = numel(setdiff(ep_indices, index));    % 误检数
FN = numel(setdiff(index, ep_indices));    % 漏检数

%% 绘制结果 - 统计图
figure;
subplot(2,1,1);
hold on;
scatter(index, f(index), 50, 'r', 'filled', 'DisplayName', '真实离群点');
scatter(ep_indices, f(ep_indices), 60, 'g', 'x', 'LineWidth', 2, 'DisplayName', '识别离群点');
xlabel('数据点索引');
ylabel('值');
title(sprintf('数据对比与离群点检测\n添加%d个离群点，准确率%.1f%%,\n漏检数%d,误检数%d\n运行时间%.2f', K, accuracy,FN,FP,t1));
legend('Location', 'best');
grid on;

subplot(2,1,2);
hold on;
% 这里修正为使用 error_vec (向量) 而不是 erros (标量)
if ~isempty(index)
    scatter(index, error_vec(index), 40, 'r', 'filled', 'DisplayName', '真实异常点误差');
end
diff_idx = setdiff(ep_indices, index);
if ~isempty(diff_idx)
    scatter(diff_idx, error_vec(diff_idx), 40, 'b', 'x', 'DisplayName', '误检点误差');
end
xlabel('数据点索引');
ylabel('误差绝对值');
title(sprintf('误差分布'));
legend('Location', 'best');
grid on;

%% 绘图部分 - xf, yf, zf 三个散点图 (修改了绿色包围圈)
figure('Position', [100, 100, 1500, 400]);

% 定义绘图点的大小
size_inner = 40;   % 实心点大小
size_outer = 80;  % 绿色包围圈大小

% 1. x-f 散点图
subplot(1,3,1);
scatter(x, f, size_inner, 'b', 'filled', 'MarkerFaceAlpha', 0.6); % 所有点
hold on;
% 绘制红色的实心异常点
scatter(x(ep_indices), f(ep_indices), size_inner, 'r', 'filled'); 
% 绘制绿色的空心包围圈 ('g'代表绿色, LineWidth加粗)
scatter(x(ep_indices), f(ep_indices), size_outer, 'g', 'LineWidth', 0.63); 

xlabel('x '); ylabel(' f');
title('x-f plane'); grid on;
legend('normal point', 'outlier point', 'outlier enclosed', 'Location', 'best');

% 2. y-f 散点图
subplot(1,3,2);
scatter(y, f, size_inner, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
scatter(y(ep_indices), f(ep_indices), size_inner, 'r', 'filled');
scatter(y(ep_indices), f(ep_indices), size_outer, 'g', 'LineWidth', 0.63); % 绿色包围圈

xlabel('y '); ylabel(' f');
title('y-f plane'); grid on;
% legend('normal point', 'outlier point',  'Location', 'best');

% 3. z-f 散点图
subplot(1,3,3);
scatter(z, f, size_inner, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
scatter(z(ep_indices), f(ep_indices), size_inner, 'r', 'filled');
scatter(z(ep_indices), f(ep_indices), size_outer, 'g', 'LineWidth', 0.63); % 绿色包围圈

xlabel('z '); ylabel(' f');
title('z-f plane'); grid on;
% legend('normal point', 'outlier point', 'Location', 'best');
