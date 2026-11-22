function [X_denoised, outliers] = MOD(X, k)
% 流形去噪与异常值检测算法
% 输入:
%   X: 原始数据矩阵 (D×N)，D是维度，N是样本数
%   k: 近邻数量
% 输出:
%   X_denoised: 去噪后的数据
%   outliers: 异常值索引

[D, N] = size(X);
X_denoised = zeros(D, N);
S_vectors = zeros(D, N); % 存储每个点的投影方向

% 步骤1: 对每个点进行局部流形拟合和投影去噪
for i = 1:N
    % 找到k个最近邻
    distances = sqrt(sum((X - X(:, i)).^2, 1));
    [~, idx] = sort(distances);
    neighbors = X(:, idx(2:k+1)); % 不包括自身
    
    % 使用PCA拟合局部流形（超平面）
    mu = mean(neighbors, 2);
    centered_neighbors = neighbors - mu;
    [V, ~] = eig(centered_neighbors * centered_neighbors');
    
    % 获取投影方向（最小特征值对应的特征向量）
    S = V(:, 1); % 假设最小特征值对应第一个特征向量
    S_vectors(:, i) = S;
    
    % 计算点到超平面的距离
    d = abs(S' * (X(:, i) - mu));
    
    % 将点投影到局部流形上
    X_denoised(:, i) = X(:, i) - d * S;
end

% 步骤2: 计算噪声并估计其分布
delta = X - X_denoised;

% 将噪声投影到S方向上进行统计分析
delta_projected = zeros(1, N);
for i = 1:N
    delta_projected(i) = S_vectors(:, i)' * delta(:, i);
end

% 使用WSD模型将分布转换为两个半正态分布
mu_noise = mean(delta_projected);
sigma_noise = std(delta_projected);

% 计算P = Pr(δ ≤ μ)
P = mean(delta_projected <= mu_noise);

% 计算两个标准差
sigma_l = P * sigma_noise;
sigma_u = (1 - P) * sigma_noise;

% 步骤3: 异常值检测
outliers = [];
for i = 1:N
    % 计算点在S方向上的投影值
    projection_value = S_vectors(:, i)' * X(:, i);
    denoised_projection = S_vectors(:, i)' * X_denoised(:, i);
    
    % 计算边界
    lower_bound = denoised_projection + mu_noise - 3 * sigma_l;
    upper_bound = denoised_projection + mu_noise + 3 * sigma_u;
    
    % 检查点是否在边界内
    if projection_value < lower_bound || projection_value > upper_bound
        outliers = [outliers, i];
    end
end

end
