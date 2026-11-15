function [f,Ik] = w_test(r,k,h)
% r  : 距离
% k  : 支撑半径系数
% h  : 局部尺度
% f  : 权重
% Ik : 是否位于支撑域内

R = k*h;

% 判断是否在支撑域内
Ik = (r >= 0) & (r < R);

% 初始化权重
f = zeros(size(r));

% 计算支撑域内的权重
f(Ik) = exp(-r(Ik).^2 ./ (r(Ik)-R).^2);

end