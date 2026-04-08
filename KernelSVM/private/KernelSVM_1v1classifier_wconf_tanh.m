function [Y_pred, confidence] = KernelSVM_1v1classifier_wconf_tanh(X_test,alpha,b,kernel,Yu,Xr_cell,yr_cell)
    nPairs = length(b);
    k = (1+sqrt(1+8*nPairs))/2;
    pairs = nchoosek(1:k,2);
    n_test = size(X_test,1);
    F = zeros(n_test,nPairs);
    confidenceVotes = zeros(size(X_test,1),k);
    for p = 1:nPairs
        K_test = kernel(Xr_cell{p},X_test);
        F(:,p) = b(p) + (alpha{p}.*yr_cell{p})'*K_test;
        n1 = pairs(p,1); n2 = pairs(p,2);
        confidenceVotes(:,n1) = confidenceVotes(:,n1) + (F(:,p)>0).*tanh(abs(F(:,p)));
        confidenceVotes(:,n2) = confidenceVotes(:,n2) + (F(:,p)<0).*tanh(abs(F(:,p)));
    end
    [~, idx_pred] = max(confidenceVotes, [], 2);
    Y_pred = Yu(idx_pred);
    totalVotes = sum(confidenceVotes, 2);
    confidence = confidenceVotes(sub2ind(size(confidenceVotes), (1:n_test)', idx_pred)) ./ totalVotes;
    confidence = confidence/max(confidence);
end