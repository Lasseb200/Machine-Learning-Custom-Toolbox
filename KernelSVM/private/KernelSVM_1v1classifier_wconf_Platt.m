function [Y_pred, confidence] = KernelSVM_1v1classifier_wconf_Platt(X_test,alpha,b,kernel,Yu,Xr_cell,yr_cell,plattA,plattB)
    nPairs = length(b);
    k = (1+sqrt(1+8*nPairs))/2;
    pairs = nchoosek(1:k,2);
    n_test = size(X_test,1);
    F = zeros(n_test,nPairs);
    votes = zeros(size(X_test,1),k);
    for p = 1:nPairs
        K_test = kernel(Xr_cell{p},X_test);
        F(:,p) = b(p) + (alpha{p}.*yr_cell{p})'*K_test;
        n1 = pairs(p,1); n2 = pairs(p,2);
        P_pos = 1 ./ (1 + exp(plattA(p)*F(:,p) + plattB(p)));  % probability that class pairs(p,1) wins
        P_neg = 1 - P_pos;
        votes(:,n1) = votes(:,n1) + P_pos;
        votes(:,n2) = votes(:,n2) + P_neg;
    end
    [~, idx_pred] = max(votes, [], 2);
    Y_pred = Yu(idx_pred);
    totalVotes = sum(votes, 2);
    confidence = votes(sub2ind(size(votes), (1:n_test)', idx_pred)) ./ totalVotes;
    confidence = confidence/max(confidence);
end