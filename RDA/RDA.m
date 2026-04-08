function [Y_pred, confidence] = RDA(X_training,Y_training,X_test,alpha)
    Ycat = categorical(Y_training);
    Yu = categories(Ycat);
    R = string(Y_training)==string(Yu)';
    k = length(Yu);
    [n,d] = size(X_training);
    n_test = size(X_test,1);
    M = grpstats(X_training, Ycat, {'mean'});
    p = grpstats(X_training, Ycat, {'numel'}); p = p(:,1)./sum(p(:,1),1);
    P = ones(n_test,1)*p';
    W_pooled = (X_training-R*M)'*(X_training-R*M);
    W_pooled = W_pooled + sqrt(eps)*eye(size(W_pooled));
    Sigma_pooled = W_pooled./(n-k);
    for j = 1:k
        X_class = X_training(Ycat == Yu(j), :);
        n_j = size(X_class,1);
        W = (X_class-M(j,:))'*(X_class-M(j,:));
        W = W + sqrt(eps)*eye(size(W));
        Sigma = W./(n_j-1);
        Sigma = alpha*Sigma + (1-alpha)*Sigma_pooled;
        
        eg = eig(Sigma); 
        eg(eg < eps) = eps;
        s(j) = d*log(2*pi) + sum(log(eg));

        Delta_j = X_test-M(j,:);
        T(:,j) = sum((Delta_j / Sigma) .* Delta_j, 2);
    end
    S = ones(n_test,1)*s;
    G = log(P) - 0.5*S - 0.5*T;
    [maxG, idx_pred] = max(G,[],2);
    Y_pred = Yu(idx_pred);
    confidence = 1 ./ sum(exp(G-maxG), 2);
end