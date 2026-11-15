% KB_function_approximation.m
% Function approximation experiment from the manuscript's Table 3.
% N=2000 paired training points, 400 paired test points, d=2,
% delta=20h, m=3, K=5%N, and 100 paired trials.

clear; clc; close all;
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(fullfile(scriptDir,'function'));

%% Settings for the synchronously paired-parameter experiment

N = 2000;
Ntest = 400;
T = 100;
K = floor(0.05*N);
epsilon = 1e-6;
d = 2;
m = 3;
noiseLevel = 0;
rng(1);

%% Synchronously paired parameter samples requested by the user.
% This samples a one-dimensional parameter curve on the Klein-bottle
% parametrization, rather than a two-dimensional tensor-product surface.
u = linspace(0,2*pi,N);
v = linspace(0,2*pi,N);

r = [(2*cos(v)+1).*cos(u);
     (2*cos(v)+1).*sin(u);
     2*sin(v).*cos(u/2);
     2*sin(v).*sin(u/2)];

fTrue = (0.5*sin(4*u) + 0.2*cos(2*v).^2 + ...
    (1/6)*cos(4*u).*sin(2*v).^2)';
fTrue = fTrue(:);

%% Fill distance and support radius
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
% The synchronously paired samples form a one-dimensional curve.  With
% d=2 and m=3, Q=10 basis functions require a wider local support than
% the manuscript's two-dimensional-surface setting delta=0.5h.
delta = 0.5*h;

%% Independent test set and exact test values
uTest = linspace(0,2*pi,Ntest);
vTest = linspace(0,2*pi,Ntest);
rTest = [(2*cos(vTest)+1).*cos(uTest);
         (2*cos(vTest)+1).*sin(uTest);
         2*sin(vTest).*cos(uTest/2);
         2*sin(vTest).*sin(uTest/2)];
fTest = (0.5*sin(4*uTest) + 0.2*cos(2*vTest).^2 + ...
    (1/6)*cos(4*uTest).*sin(2*vTest).^2)';
fTest = fTest(:);

%% Construct the reusable Manifold-MLS matrix
tic;
[A,geometry] = MMLSforA(r,epsilon,d,m,h,delta);
buildTime = toc;
U_mls = speye(N)-sparse(A);
UN_mls = sqrt(sum(U_mls.^2,1))';
fprintf('Klein bottle function approximation: N=%d, Ntest=%d, K=%d\n', ...
    N,Ntest,K);
fprintf('h=%.8g, delta=%.8g, A-build time=%.3f s\n',h,delta,buildTime);

%% Evaluate only after A has been built from the 2000 training points.
% The 400 points are independent test locations. They are not used to
% construct A and their exact values are never passed to MMLSforoutlier.
[fCleanApprox,Btest] = MMLSApproximate(r,rTest,fTrue, ...
    epsilon,d,m,h,delta);
[cleanL2,cleanRMSE] = approximationErrors(fCleanApprox,fTest);

%% Paired corrupted and recovered trials
corruptedL2 = nan(T,1);
corruptedRMSE = nan(T,1);
recoveredL2 = nan(T,1);
recoveredRMSE = nan(T,1);
recoveryFailed = false(T,1);

for trial = 1:T
    index = sort(randperm(N,K));
    E = zeros(N,1);
    E(index) = (2*randi([0,1],K,1)-1).*(0.6+0.5*rand(K,1));

    % The manuscript's Table 3 compares outlier contamination and recovery
    % without adding the separate 2% noise used in the robustness table.
    fCorrupted = fTrue + E;
    fRecovered = fCorrupted;

    try
        [Ep,~,~] = MMLSforoutlier(A,fCorrupted,U_mls,UN_mls);
        fRecovered = fCorrupted - Ep(:);
    catch ME
        recoveryFailed(trial) = true;
        fprintf(2,'Trial %d recovery failed: %s\n',trial,ME.message);
    end

    % A detects/recoveries are performed on the 2000 training values.
    % The recovered training values are then evaluated at the 400 tests.
    sCorrupted = Btest*fCorrupted;
    sRecovered = Btest*fRecovered;

    [corruptedL2(trial),corruptedRMSE(trial)] = ...
        approximationErrors(sCorrupted,fTest);
    [recoveredL2(trial),recoveredRMSE(trial)] = ...
        approximationErrors(sRecovered,fTest);
end

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
    'recoveredL2','recoveredRMSE','recoveryFailed','A','geometry', ...
    'r','rTest','fTrue','fTest','h','delta','N','Ntest','K','T');

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

