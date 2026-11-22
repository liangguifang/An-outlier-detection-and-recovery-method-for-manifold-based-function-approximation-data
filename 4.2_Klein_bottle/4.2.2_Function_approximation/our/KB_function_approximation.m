% Klein bottle prediction before and after outlier recovery.
clear; clc; close all;
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(fullfile(scriptDir,'function'));

%% 与 KLB100 配对的训练及异常设置
N = 2000;
Ntest = 400;
T = 100;
K = floor(0.05*N);
d = 2;
m = 3;
geometrySeed = 1;
testSeed = 2;
ompTau = 0.5;
outlierMagnitude = [0.6,1.1];

[r,u,v,fTrue] = MMLSKleinBottleSamples(N,geometrySeed);
[rTest,uTest,vTest,fTest] = MMLSKleinBottleSamples(Ntest,testSeed);
outlierStream = RandStream('mt19937ar','Seed',1);

%% 一次构造训练矩阵 A 和训练到测试的预测矩阵 Btest
buildTimer = tic;
[A,neighborCount,localRadii,geometry,Btest] = ...
    buildLocalPcaMLS(r.',d,m,[],true,false,rTest.');
buildTime = toc(buildTimer);
U_mls = speye(N)-A;
UN_mls = vecnorm(U_mls,2,1)';
delta = max(localRadii);
omega_delta = geometry.omega_delta;

fprintf('Klein bottle function approximation: N=%d, Ntest=%d, K=%d\n', ...
    N,Ntest,K);
fprintf('sigma=%.8g, local support radii=[%.8g, %.8g], A/Btest build time=%.3f s\n', ...
    geometry.sigma,min(localRadii),delta,buildTime);
fprintf('OMP tau=%.6g\n',ompTau);

% 测试点真值仅用于误差评估。
fCleanApprox = Btest*fTrue;
[cleanL2,cleanRMSE] = approximationErrors(fCleanApprox,fTest);

%% Paired corrupted and recovered trials
corruptedL2 = nan(T,1);
corruptedRMSE = nan(T,1);
recoveredL2 = nan(T,1);
recoveredRMSE = nan(T,1);
recoveryFailed = false(T,1);
outlierIndices = zeros(K,T);
outlierValues = zeros(K,T);

for trial = 1:T
    [index,E] = MMLSSignedOutliers(outlierStream,N,K,outlierMagnitude);
    outlierIndices(:,trial) = index(:);
    outlierValues(:,trial) = E(index);

    % The manuscript's Table 3 compares outlier contamination and recovery
    % without adding the separate 2% noise used in the robustness table.
    fCorrupted = fTrue + E;
    fRecovered = fCorrupted;

    try
        [Ep,~,~] = MMLSforoutlier(A,fCorrupted,U_mls,UN_mls,ompTau);
        fRecovered = fCorrupted - Ep(:);
    catch ME
        recoveryFailed(trial) = true;
        fprintf(2,'Trial %d recovery failed: %s\n',trial,ME.message);
    end

    % 在训练点上恢复函数值，再使用固定的 Btest 预测测试点。
    sCorrupted = Btest*fCorrupted;
    sRecovered = Btest*fRecovered;

    [corruptedL2(trial),corruptedRMSE(trial)] = ...
        approximationErrors(sCorrupted,fTest);
    [recoveredL2(trial),recoveredRMSE(trial)] = ...
        approximationErrors(sRecovered,fTest);
end

%% 与 KLB100 一致，试验结束后单独计算训练样本的投影分离距离
Q0_all = nan(T,1);
for trial = 1:T
    Q0_all(trial) = MMLSProjectedSeparation(geometry,outlierIndices(:,trial));
end
Q0_mean = mean(Q0_all,'omitnan');
Q0_std = std(Q0_all,'omitnan');
Q0Condition_all = delta + omega_delta <= Q0_all;
fprintf('q_tilde_0 = %.6e +/- %.6e, separation condition = %d / %d trials\n', ...
    Q0_mean,Q0_std,sum(Q0Condition_all),T);

%% Report and save the manuscript-style table
valid = ~recoveryFailed;
resultTable = table( ...
    ["Clean";"Corrupted";"Recovered"], ...
    [cleanL2;mean(corruptedL2(valid));mean(recoveredL2(valid))], ...
    [NaN;std(corruptedL2(valid));std(recoveredL2(valid))], ...
    [cleanRMSE;mean(corruptedRMSE(valid));mean(recoveredRMSE(valid))], ...
    [NaN;std(corruptedRMSE(valid));std(recoveredRMSE(valid))], ...
    'VariableNames',{'Data','RelativeL2Mean','RelativeL2Std', ...
    'RMSEMean','RMSEStd'});

outDir = fullfile(scriptDir,'KB_function_approximation_output');
if ~isfolder(outDir), mkdir(outDir); end
writetable(resultTable,fullfile(outDir, ...
    'KB_function_approximation_results.csv'));
save(fullfile(outDir,'KB_function_approximation_results.mat'), ...
    'resultTable','cleanL2','cleanRMSE','corruptedL2','corruptedRMSE', ...
    'recoveredL2','recoveredRMSE','recoveryFailed','A','Btest','geometry', ...
    'r','rTest','u','v','uTest','vTest','fTrue','fTest', ...
    'localRadii','neighborCount','delta','omega_delta','N','Ntest','K','T','buildTime', ...
    'geometrySeed','testSeed','ompTau','outlierIndices','outlierValues', ...
    'Q0_all','Q0_mean','Q0_std','Q0Condition_all');

fprintf('\nData                 Relative L2              RMSE\n');
fprintf('Clean                %.6f                    %.6f\n',cleanL2,cleanRMSE);
fprintf('Corrupted            %.6f +/- %.6f        %.6f +/- %.6f\n', ...
    mean(corruptedL2(valid)),std(corruptedL2(valid)), ...
    mean(corruptedRMSE(valid)),std(corruptedRMSE(valid)));
fprintf('Recovered            %.6f +/- %.6f        %.6f +/- %.6f\n', ...
    mean(recoveredL2(valid)),std(recoveredL2(valid)), ...
    mean(recoveredRMSE(valid)),std(recoveredRMSE(valid)));

fig = figure('Color','w','Position',[100 100 900 600]);
meansL2 = resultTable.RelativeL2Mean;
stdsL2 = resultTable.RelativeL2Std;
meansRMSE = resultTable.RMSEMean;
stdsRMSE = resultTable.RMSEStd;
stdsL2(1) = 0;
stdsRMSE(1) = 0;
subplot(1,2,1);
bar(meansL2);
hold on;
errorbar((1:3)',meansL2,zeros(3,1),stdsL2,'.k','LineWidth',1.2);
set(gca,'XTick',1:3,'XTickLabel',{'Clean','Corrupted','Recovered'});
ylabel('Relative L_2 error'); grid on; box on;
subplot(1,2,2);
bar(meansRMSE);
hold on;
errorbar((1:3)',meansRMSE,zeros(3,1),stdsRMSE,'.k','LineWidth',1.2);
set(gca,'XTick',1:3,'XTickLabel',{'Clean','Corrupted','Recovered'});
ylabel('RMSE'); grid on; box on;
sgtitle(sprintf('Klein bottle function approximation, N=%d',N));

print(fig,fullfile(outDir,'KB_function_approximation_errors.png'),'-dpng','-r300');
if exist('exportgraphics','file') == 2
    exportgraphics(fig,fullfile(outDir, ...
        'KB_function_approximation_errors.pdf'),'ContentType','vector');
end
savefig(fig,fullfile(outDir,'KB_function_approximation_errors.fig'));

function [relativeL2,rmse] = approximationErrors(approximation,truth)
difference = approximation(:)-truth(:);
truth = truth(:);
relativeL2 = norm(difference,2)/norm(truth,2);
rmse = sqrt(mean(difference.^2));
end
