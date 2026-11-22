function [r,u,v,fTrue] = MMLSKleinBottleSamples(N,geometrySeed)
% Draw independent uniform parameter pairs on [0,2*pi)^2.
geometryStream = RandStream('mt19937ar','Seed',geometrySeed);
u = 2*pi*rand(geometryStream,1,N);
v = 2*pi*rand(geometryStream,1,N);

x1 = (2.*cos(v)+1).*cos(u);
x2 = (2.*cos(v)+1).*sin(u);
x3 = 2.*sin(v).*cos(u./2);
x4 = 2.*sin(v).*sin(u./2);
r = [x1;x2;x3;x4];

fTrue = ( ...
    1/2 .* sin(4 .* u) ...
    + 1/5 .* cos(2 .* v).^2 ...
    + 1/6 .* cos(4 .* u).*sin(2 .* v).^2 ...
    )';
end
