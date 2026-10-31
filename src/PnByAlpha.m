function p=PnByAlpha(X,Alpha)

[~,d]=size(Alpha);
if length(X)~=d
    error('����X����')
else

    x=X(:)';

    p= prod(x.^Alpha, 2)';

end
end
