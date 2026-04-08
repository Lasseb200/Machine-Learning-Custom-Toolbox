function [Y_pred, confidence, probabilities, class_labels] = NB(X_training,Y_training,X_test)
    Ycat = categorical(Y_training);
    Yu = categories(Ycat);
    k = length(Yu);
    n_test = size(X_test,1);
    d = size(X_test,2);
    M = grpstats(X_training, Ycat, {'mean'});
    V = grpstats(X_training, Ycat, {'var'});
    p = grpstats(X_training, Ycat, {'numel'}); p = p(:,1)./sum(p(:,1),1);
    P = ones(n_test,1)*p';
    s = sum(log(2*pi.*V),2);
    S = ones(n_test,1)*s';
    M_n = reshape(M,[1, k, d]); M_n = repmat(M_n,[n_test, 1, 1]);
    X_test_k = reshape(X_test,[n_test,1,d]); X_test_k = repmat(X_test_k,[1 k 1]);
    V_n = reshape(V,[1, k, d]); V_n = repmat(V_n,[n_test, 1, 1]);
    G = log(P)-1/2.*S-1/2.*sum((X_test_k-M_n).^2./V_n,3);
    [maxG, idx_pred] = max(G,[],2);
    Y_pred = string(Yu(idx_pred));
    probabilities = exp(G-maxG)./sum(exp(G-maxG),2);
    class_labels = string(Yu);
    confidence = max(probabilities,[],2);
end