function [Y_pred, confidence] = KernelSVM_1v1classifier_wconf(X_test,alpha,b,kernel,Yu,Xr_cell,yr_cell)
    nPairs = length(b);
    k = (1+sqrt(1+8*nPairs))/2;
    pairs = nchoosek(1:k,2);
    n_test = size(X_test,1);
    votes = zeros(size(X_test,1),k); F = zeros(n_test,nPairs);
    for p = 1:nPairs
        K_test = kernel(Xr_cell{p},X_test);
        F(:,p) = b(p) + (alpha{p}.*yr_cell{p})'*K_test;
        n1 = pairs(p,1); n2 = pairs(p,2);
        votes(:,n1) = votes(:,n1) + (F(:,p)>0);
        votes(:,n2) = votes(:,n2) + (F(:,p)<0);
    end
    [~, idx_pred] = max(votes, [], 2);
    Y_pred = Yu(idx_pred);
    totalVotes = sum(votes, 2);
    confidence = votes(sub2ind(size(votes), (1:n_test)', idx_pred)) ./ totalVotes;
end