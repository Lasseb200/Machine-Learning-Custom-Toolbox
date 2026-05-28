function [v, lambda, B, W] = LDA(X_training,Y_training,n)
    Yu = unique(Y_training,'stable');
    R = Y_training==Yu';
    d = size(X_training,2);
    k = size(R,2);
    n_class(:,1) = sum(R,1);
    M = R'*X_training./(n_class*ones(1,d));
    mu_bar(:,1) = mean(X_training,1);
    D = M-ones(k,1)*mu_bar';
    N = diag(n_class);
    B = D'*N*D;
    W = (X_training-R*M)'*(X_training-R*M);
    alpha = 0.002+0.0001*d;
    %alpha = 1e-4;
    %alpha = 0.05;
    % n_training = size(X_training,1);
    % ratio = n_training/d;
    % alpha = 0.1*ratio;
    W = W + alpha * trace(W)/size(W,1) * eye(size(W));
    fprintf('alpha: %f, cond: %f\n', alpha, cond(W));
    [v, lambda] = eigs(B,W,n);
    v = real(v);
    lambda = diag(lambda);
end