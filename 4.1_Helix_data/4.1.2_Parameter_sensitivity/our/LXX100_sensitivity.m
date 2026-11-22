% Helix parameter sensitivity (paired outlier trials).
clear; clc; close all;
scriptDir=fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir,'function'));

N=1000; T=100; K=floor(0.05*N);
d=1;
baseK=1.5; baseM=3; baseEps=0.5;
kValues=[0.5 1 1.5 2 3];
outlierMagnitude=[0.05,0.10];
mValues=[1 2 3 4];
epsValues=[0.05 0.10 0.20 0.30 0.50 0.55 0.70];

t=linspace(-pi,pi,N);
r=[sin(t);cos(t);t];
fTrue=frankgaowei(r)'; fTrue=fTrue(:);
[baseA,~,~,baseGeometry]=buildLocalPcaMLS(r.',d,baseM,[],true,false);
baseSigma=baseGeometry.sigma;
baseEtaFactor=abs(median(fTrue));
% Geometric, non-equally-spaced sampling around the median-based reference.
etaMultipliers=[0.01 0.03 0.1 0.3 1 3 10 30 50];
etaValues=baseEtaFactor*etaMultipliers;
if baseEtaFactor <= 0 || ~isfinite(baseEtaFactor)
    error('The median-based eta reference must be positive for logarithmic sampling.');
end
outlierStream=RandStream('mt19937ar','Seed',1);
ETrue=zeros(N,T); FObserved=zeros(N,T);
for trial=1:T
    [~,e]=MMLSSignedOutliers(outlierStream,N,K,outlierMagnitude);
    ETrue(:,trial)=e;
    FObserved(:,trial)=fTrue+e; % 仅添加异常值，不添加小噪声。
end
trialEtaFactors=abs(median(FObserved,1));

names={'k','m','etaFactor','Eps'};
sets={kValues,mValues,etaValues,epsValues};
nCases=sum(cellfun(@numel,sets));
template=struct('Parameter','','Value',NaN,'k',NaN,'m',NaN, ...
 'EtaFactor',NaN,'Eps',NaN,'Status','FAILED','F1Mean',NaN, ...
 'F1Std',NaN,'RecallMean',NaN,'PrecisionMean',NaN,'ErrorMean',NaN);
results=repmat(template,nCases,1); row=0;
rawF1=nan(nCases,T);

for g=1:4
 for value=sets{g}
  row=row+1; p=template;
  p.Parameter=names{g}; p.Value=value; p.k=baseK; p.m=baseM;
  p.EtaFactor=baseEtaFactor; p.Eps=baseEps;
  switch names{g}
   case 'k', p.k=value;
   case 'm', p.m=value;
   case 'etaFactor', p.EtaFactor=value;
   case 'Eps', p.Eps=value;
  end
  try
   if p.k==baseK && p.m==baseM
    A=baseA;
   else
    % k_2 乘以最远近邻距离。固定邻域和 FLCS 半径，只改变拟合半径。
    fitRadiusScale=p.k/baseK;
    A=buildLocalPcaMLS(r.',d,p.m,[],true,false,[],baseSigma,fitRadiusScale);
   end
   U=speye(N)-A; UN=vecnorm(U,2,1)'; Um=min(UN);
   % 横轴仍显示由干净中位数标定的名义值，参考组逐轮使用观测中位数。
   etaScale=p.EtaFactor/baseEtaFactor;
   scores=nan(T,4);
   for trial=1:T
    eta=(Um*trialEtaFactors(trial))*etaScale;
    Ep=MMLSforoutlier(A,FObserved(:,trial),U,UN,p.Eps,eta);
    truth=ETrue(:,trial)~=0;
    detected=abs(Ep)>1e-12;
    TP=nnz(truth&detected); FP=nnz(~truth&detected); FN=nnz(truth&~detected);
    rec=TP/K; prec=0; f1=0;
    if TP+FP>0, prec=TP/(TP+FP); end
    if 2*TP+FP+FN>0, f1=2*TP/(2*TP+FP+FN); end
    scores(trial,:)=[rec,prec,f1,norm(ETrue(:,trial)-Ep,2)/K];
   end
   p.Status='OK'; p.F1Mean=mean(scores(:,3)); p.F1Std=std(scores(:,3));
   p.RecallMean=mean(scores(:,1)); p.PrecisionMean=mean(scores(:,2));
   p.ErrorMean=mean(scores(:,4));
   rawF1(row,:)=scores(:,3)';
  catch ME
   fprintf(2,'[%s=%g] %s\n',names{g},value,ME.message);
  end
  results(row)=p;
 end
end

outDir=fullfile(scriptDir,'LXX100_sensitivity_output');
if ~isfolder(outDir), mkdir(outDir); end
writetable(struct2table(results),fullfile(outDir,'LXX100_sensitivity_results.csv'));
save(fullfile(outDir,'LXX100_sensitivity_results.mat'),'results','rawF1','N','K','T', ...
 'baseK','baseSigma','baseM','baseEtaFactor','baseEps','kValues','mValues','etaValues','etaMultipliers','epsValues');

baseValues=[baseK baseM baseEtaFactor baseEps];
fig=figure('Color','w');
axisLabels={'$k_2$','$m$','$\eta$ factor','$\tau$'};
for g=1:4
 subplot(2,2,g);
 use=strcmp({results.Parameter},names{g})&strcmp({results.Status},'OK');
 rr=results(use);
 xScale=1; xLabel=axisLabels{g};
 if g==3
  xScale=1;
  xLabel='$\gamma$';
 end
 errorbar(xScale*[rr.Value],[rr.F1Mean],[rr.F1Std],'-o', ...
  'LineWidth',1.3,'CapSize',8,'DisplayName','Mean F1-score \pm SD');
 hold on;
 referenceIndex=find([rr.Value]==baseValues(g),1);
 if ~isempty(referenceIndex)
  plot(xScale*rr(referenceIndex).Value,rr(referenceIndex).F1Mean,'r^', ...
   'MarkerSize',13,'MarkerFaceColor','r','DisplayName','Reference');
 end
 if g==3
  set(gca,'XScale','log','TickLabelInterpreter','tex');
  etaTicks=baseEtaFactor*[0.01 0.1 1 10 50];
  xticks(etaTicks);
  xticklabels(compose('%.4g',etaTicks));
  xlim([min(etaValues) max(etaValues)]);
 else
  xticks(sets{g});
  xticklabels(compose('%.3g',sets{g}));
  xlim([min(sets{g}) max(sets{g})]);
  if g==4
   % Leave space beyond the endpoint so MATLAB displays the final label.
   tauMargin=0.03*(max(epsValues)-min(epsValues));
   xlim([min(epsValues)-tauMargin max(epsValues)+tauMargin]);
   xticks(epsValues);
   xticklabels(compose('%.3g',epsValues));
  end
 end
 ylim([0 1.05]);
 yticks([0 0.2 0.4 0.6 0.8 1]);
 grid on; box on;
 if g==3
  set(gca,'XMinorGrid','off','XMinorTick','off');
 end
 xlabel(xLabel,'Interpreter','latex'); ylabel('F1-score');
 if g==1
  legend('Location','best','Interpreter','tex');
 end
end
sgtitle(sprintf('Helix sensitivity, N=%d, K=%d, trials=%d',N,K,T));
print(fig,fullfile(outDir,'LXX100_sensitivity_F1.png'),'-dpng','-r300');
if exist('exportgraphics','file')==2
 exportgraphics(fig,fullfile(outDir,'LXX100_sensitivity_F1.pdf'),'ContentType','vector');
end
if isgraphics(fig,'figure'), savefig(fig,fullfile(outDir,'LXX100_sensitivity_F1.fig')); end

% Additional box plots of the same trials. Parameter settings are displayed
% as equally spaced groups, with their actual values on the tick labels.
boxFig=figure('Color','w','Position',[100 100 1100 760]);
boxLabels={'$k_2$','$m$','$\gamma$','$\tau$'};
for g=1:4
 subplot(2,2,g);
 use=strcmp({results.Parameter},names{g})&strcmp({results.Status},'OK');
 rr=results(use);
 groupF1=rawF1(use,:);
 if ~isempty(rr)
  groupPositions=repelem((1:numel(rr))',T);
  groupData=reshape(groupF1',[],1);
  boxchart(groupPositions,groupData,'DisplayName','F1-score distribution');
  hold on;
  referenceIndex=find([rr.Value]==baseValues(g),1);
  if ~isempty(referenceIndex)
   plot(referenceIndex,median(groupF1(referenceIndex,:)),'r^', ...
    'MarkerSize',13,'MarkerFaceColor','r','DisplayName','Reference');
  end
  tickValues=[rr.Value];
  xticks(1:numel(rr));
  xticklabels(compose('%.3g',tickValues));
  xlim([0.5 numel(rr)+0.5]);
  if g==1
   legend('Location','best');
  end
 end
 ylim([0 1.05]);
 yticks([0 0.2 0.4 0.6 0.8 1]);
 grid on; box on;
 xlabel(boxLabels{g},'Interpreter','latex'); ylabel('F1-score');
end
sgtitle(sprintf('Helix F1-score distributions, N=%d, K=%d, trials=%d',N,K,T));
print(boxFig,fullfile(outDir,'LXX100_sensitivity_boxplot.png'),'-dpng','-r300');
if exist('exportgraphics','file')==2
 exportgraphics(boxFig,fullfile(outDir,'LXX100_sensitivity_boxplot.pdf'), ...
  'ContentType','vector');
end
if isgraphics(boxFig,'figure')
 savefig(boxFig,fullfile(outDir,'LXX100_sensitivity_boxplot.fig'));
end
