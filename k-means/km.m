function [Y_pred_idx, confidence, M, beta] = km(X,k)
    n = size(X,1);
    d = size(X,2);
    idx = randperm(n,k);
    M = X(idx,:);
    iter = 1;
    while true
    %for kkk = 1:1
        D = pdist2(X,M,'squaredeuclidean');
        [~, idx] = min(D,[],2);
        R = zeros(n,k);
        R(sub2ind([n, k], (1:n)',idx)) = 1;
        M_new = (R'*X)./((ones(1,n)*R)'*ones(1,d));
        if abs(mean(M_new(:)-M(:)))<1e-8
            break;
        end
        M = M_new;
        iter = iter+1;
    end
    %[~, Y_pred_idx] = max(R,[],2);
    %beta = 5;
    %D = D-min(D,2);
    %P = exp(-beta.*D)./(sum(exp(-beta.*D),2));
    %confidence = max(P,[],2);
    [~, Y_pred_idx] = max(R,[],2);
    sigma_sq = zeros(k,1);
    for j = 1:k
        Xj = X(Y_pred_idx == j,:);
        Mj = M(j,:);
        dist_j = pdist2(Xj,Mj,'squaredeuclidean');
        variance(j) = mean(dist_j);
    end
    beta = 1./variance;
    D = pdist2(X,M,'squaredeuclidean');
    D = D - min(D,[],2);
    P = exp(-beta.*D) ./ sum(exp(-beta.*D),2);
    confidence = max(P,[],2);
end