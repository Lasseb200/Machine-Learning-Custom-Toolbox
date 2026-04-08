function [v, lambda] = PCA(X)
    n = size(X,1);
    d = size(X,2);
    mu = mean(X,1);
    B = X-ones(n,1)*mu;
    C = (B'*B)/n;
    [v, lambda] = eigs(C,d);
    lambda = diag(lambda);
end