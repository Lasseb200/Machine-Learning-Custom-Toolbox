function [v, lambda, B, W] = OLDA(X_training,Y_training,n)
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
    cond(W)
    %W = W + sqrt(eps)*eye(size(W));
    alpha = 4e-1;   % or 1e-4 to 1e-2 
    W = W + alpha * trace(W)/size(W,1) * eye(size(W));
    cond(W)
    [v, lambda] = eigs(B,W,1);
    W_inv_B = W\B;
    if n>1
        for i = 2:n
            W_inv_v = W\v;
            S = v'*W_inv_v;
            M = (eye(d) - W_inv_v*(S\v'))*W_inv_B;
            [v(:,i), ~] = eigs(M,1);
        end
    end
    v = real(v);
end
