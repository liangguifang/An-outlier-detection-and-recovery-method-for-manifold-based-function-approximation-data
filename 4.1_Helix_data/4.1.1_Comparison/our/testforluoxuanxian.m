addpath('./function')
N=1000;
%% 生成螺旋线数据
t = linspace(-pi, pi, N);
x = sin(t);
y = cos(t);
z = t;
%%
x_norm= (x - min(x)) / (max(x) - min(x));
y_norm = (y - min(y)) / (max(y) - min(y));
z_norm = (z - min(z)) / (max(z) - min(z));
r= [x_norm; y_norm ; z_norm]; % 3×100矩阵
%r= [x; y; z];

%填充距离
N_ref = 40000;   % 稠密参考点

t_ref = linspace(-pi, pi, N_ref);
x_ref = sin(t_ref);
y_ref = cos(t_ref);
z_ref = t_ref;
x_ref_norm = (x_ref - min(x_ref)) / (max(x_ref) - min(x_ref));
y_ref_norm= (y_ref - min(y_ref)) / (max(y_ref) - min(y_ref));
z_ref_norm= (z_ref - min(z_ref)) / (max(z_ref) - min(z_ref));
M_ref_norm = [x_ref_norm; y_ref_norm; z_ref_norm];
%M_ref = [x_ref; y_ref; z_ref];

h=fillh(r,M_ref_norm);


K = floor(0.05*N);    % 异常值数量（5%的数据点）

E = zeros(N, 1);      % 初始化偏差向量

f= frankgaowei(r)';


%% 生成随机异常值索引（不重复）
index = randperm(N, K);  % 从1到N中随机选取K个不重复的索引
index = sort(index);
%% 添加小噪声
%noise1 =0.02 * rand(size(f));  % 生成与 f 相同大小的均匀噪声矩阵，噪声值在 [0, 0.5] 之间
%f = f.*(1+ noise1);          % 将噪声添加到 f
%% 添加随机符号和大小的异常值
for j = 1:K
   noise =0.5+0.5*rand(1);   
   direction = (randi([0, 1]) * 2) - 1;  % 生成随机符号：结果为 1 或 -1
   E(index(j)) = direction * noise;      % 应用带符号的异常值
end
f = f + E;  % 生成含异常值的数据（f为观测值向量）
epsilon = 1e-6;  % 收敛阈值
d=1;m=3;
tic;

% 第一部分：计算投影和MLS矩阵A
A = MMLSforA(r,epsilon,d,m,h);

% 第二部分：异常值识别
[Ep,Index,f_approx] = MMLSforoutlier(A,f);

erros1=abs(f-E-f_approx);
erros=norm(E-Ep,2)/K;
t1=toc;
%% 计算检测准确率
ep_indices = find(Ep ~= 0);  % 检测到的异常点索引
TP = numel(intersect(index, ep_indices));  % 正确检测数
accuracy = TP / K * 100;      % 准确率
FP = numel(setdiff(ep_indices, index));    % 误检数
FN=numel(setdiff(index, ep_indices)); %漏检数


%% 绘制结果
% figure;
% subplot(2,1,1);
% hold on;
% % plot(z, 'b', 'LineWidth', 1.5, 'DisplayName', '真实数据');
% % plot(f, 'm.', 'MarkerSize', 8, 'DisplayName', '含异常值观测数据');
% scatter(index, f(index), 50, 'r', 'filled', 'DisplayName', '真实离群点');
% scatter(ep_indices, f(ep_indices), 60, 'g', 'x', 'LineWidth', 2, 'DisplayName', '识别离群点');
% xlabel('数据点索引');
% ylabel('值');
% title(sprintf('数据对比与离群点检测\n添加%d个离群点，准确率%.1f%%,\n漏检数%d,误检数%d\n运行时间%.2f', K, accuracy,FN,FP,t));
% legend('Location', 'best');
% grid on;
% 
% subplot(2,1,2);
% hold on;
% scatter(index, erros(index), 40, 'r', 'filled', 'DisplayName', '真实异常点误差');
% scatter(setdiff(ep_indices, index), erros(setdiff(ep_indices, index)), 40, 'b', 'x', 'DisplayName', '误检点误差');
% xlabel('数据点索引');
% ylabel('误差绝对值');
% title(sprintf('误差分布'));
% legend('Location', 'best');
% grid on;

