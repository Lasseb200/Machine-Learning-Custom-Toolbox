function [Y_pred, confidence, probabilities, class_labels] = QDA(X_training,Y_training,X_test)
    Ycat = categorical(Y_training);
    Yu = categories(Ycat);
    R = string(Y_training)==string(Yu)';
    k = length(Yu);
    d = size(X_training,2);
    n_test = size(X_test,1);
    M = grpstats(X_training, Ycat, {'mean'});
    p = grpstats(X_training, Ycat, {'numel'}); p = p(:,1)./sum(p(:,1),1);
    P = ones(n_test,1)*p';
    for j = 1:k
        X_class = X_training(Ycat == Yu(j), :);
        n_j = size(X_class,1);
        W = (X_class-M(j,:))'*(X_class-M(j,:));
        %W = W + sqrt(eps)*eye(size(W));
        %alpha = 0.75;
        %alpha = 1e-4;
        %alpha = 0.01;
        alpha = 0.002+0.0001*d;
        W = W + alpha * trace(W)/size(W,1) * eye(size(W));
        Sigma = W./(n_j-1);

        eg = eig(Sigma); 
        eg(eg < eps) = eps;
        s(j) = sum(log(eg));
        
        Delta_j = X_test-M(j,:);
        T(:,j) = sum((Delta_j / Sigma) .* Delta_j, 2);
    end
    S = ones(n_test,1)*s;
    G = log(P) - 0.5*S - 0.5*T;
    [maxG, idx_pred] = max(G,[],2);
    Y_pred = string(Yu(idx_pred));
    probabilities = exp(G-maxG)./sum(exp(G-maxG),2);
    class_labels = string(Yu);
    confidence = max(probabilities,[],2);
end