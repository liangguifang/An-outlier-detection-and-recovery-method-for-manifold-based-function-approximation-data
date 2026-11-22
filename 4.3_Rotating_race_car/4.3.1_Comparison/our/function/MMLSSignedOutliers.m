function [index,E] = MMLSSignedOutliers(stream,N,K,magnitudeRange)
% Same location draw and interleaved magnitude/sign draws for paired experiments.
index = sort(randperm(stream,N,K));
E = zeros(N,1);
amplitudeWidth = diff(magnitudeRange);
for j = 1:K
    magnitude = magnitudeRange(1) + amplitudeWidth*rand(stream,1);
    direction = 2*randi(stream,[0,1])-1;
    E(index(j)) = direction*magnitude;
end
end
