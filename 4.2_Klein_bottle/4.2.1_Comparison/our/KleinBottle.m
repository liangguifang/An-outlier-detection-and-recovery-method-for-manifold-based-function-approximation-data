addpath('./function')
% 定义参数范围
% u = linspace(0, 2*pi, N);
% v = linspace(0, 2*pi, N);

% %1000个点
Nu = 25;
Nv =40;

% %2000个点
% Nu = 40;
% Nv =50;

% %4000个点
% Nu = 50;
% Nv =80;

u0=(0:Nu-1)*(2*pi/Nu);
v0=(0:Nv-1)*(2*pi/Nv);

[u,v]=meshgrid(u0,v0);

u=u(:)';
v=v(:)';


% 总数据点数
N = length(u);       % N = 1024

% 克莱因瓶的参数方程
x1=(2.*cos(v)+1).*cos(u);
x2=(2.*cos(v)+1).*sin(u);
x3=2.*sin(v).*cos(u./2);
x4=2.*sin(v).*sin(u./2);
r=[x1;x2;x3;x4];

%计算填充距离h
Nu_ref = 200;
Nv_ref = 200;

u_ref = linspace(0,2*pi,Nu_ref);
v_ref = linspace(0,2*pi,Nv_ref);

[uu,vv] = meshgrid(u_ref,v_ref);

uu=uu(:)';
vv=vv(:)';

x1=(2.*cos(vv)+1).*cos(uu);
x2=(2.*cos(vv)+1).*sin(uu);
x3=2.*sin(vv).*cos(uu./2);
x4=2.*sin(vv).*sin(uu./2);

M_ref=[x1;x2;x3;x4];

h=fillh(r,M_ref);



f = 1/2 .* sin(4 .* u) + 1/5 .* cos(2 .* v).^2 + 1/6 .*cos(4 .* u).*sin(2 .* v).^2;
K = floor(0.05*N);    % 异常值数量（1%的数据点）
E = zeros(N, 1);      % 初始化偏差向量

%% 生成随机异常值索引（不重复）
index = randperm(N, K);  % 从1到N中随机选取K个不重复的索引
index = sort(index);
%
% noise1 = 0.018 * rand(size(f));  % 生成与 f 相同大小的均匀噪声矩阵，噪声值在 [0, 0.5] 之间
% f = f + noise1;          % 将噪声添加到 f
%% 添加随机符号和大小的异常值
for j = 1:K
   noise = 0.6+ 0.5 * rand(1);          % 生成 0.6 到 1.5 之间的随机幅度
   direction = (randi([0, 1]) * 2) - 1;  % 生成随机符号：结果为 1 或 -1
   E(index(j)) = direction * noise;      % 应用带符号的异常值
end
f =f'+ E;  % 生成含异常值的数据（f为观测值向量）
epsilon = 1e-6;  % 收敛阈值
d=2;m=3;
tic;
%[Ep,Index]=MMLSforL0(r,f,epsilon,d,m);
% 第一部分：计算投影和MLS矩阵A
A = MMLSforA(r,epsilon,d,m,h);
t=toc;
% 第二部分：异常值识别
[Ep,Index,f_approx] = MMLSforoutlier(A,f);

erros=norm(E-Ep,2)/K;

ep_indices = find(Ep ~= 0);  % 检测到的异常点索引
TP = numel(intersect(index, ep_indices));  % 正确检测数
accuracy = TP / K * 100;      % 准确率
FP = numel(setdiff(ep_indices, index));    % 误检数
FN=numel(setdiff(index, ep_indices)); %漏检数

