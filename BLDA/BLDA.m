function [Y_pred, confidence, probabilities, class_labels] = BLDA(X_training,Y_training,X_test)
    Ycat = categorical(Y_training);
    Yu = categories(Ycat);
    R = string(Y_training)==string(Yu)';
    k = length(Yu);
    n = size(X_training,1);
    n_test = size(X_test,1);
    M = grpstats(X_training, Ycat, {'mean'});
    p = grpstats(X_training, Ycat, {'numel'}); p = p(:,1)./sum(p(:,1),1);
    P = ones(n_test,1)*p';
    W = (X_training-R*M)'*(X_training-R*M);
    %alpha = 1e-4;
    %alpha = 0.75;
    %alpha = 0.01;
    d = size(X_training,2);
    alpha = 0.002+0.0001*d;
    W = W + alpha * trace(W)/size(W,1) * eye(size(W));
    fprintf('alpha: %f, cond: %f\n', alpha, cond(W));

    W = W + alpha * trace(W)/size(W,1) * eye(size(W));
    Sigma = W./(n-k);
    Sigma_M = Sigma\M';
    T = diag(M*(Sigma_M)); T = ones(n_test,1)*T';
    G = log(P) + X_test*(Sigma_M) - 0.5.*T;
    [maxG, idx_pred] = max(G,[],2);
    Y_pred = string(Yu(idx_pred));
    probabilities = exp(G-maxG)./sum(exp(G-maxG),2);
    class_labels = string(Yu);
    confidence = max(probabilities,[],2);
end