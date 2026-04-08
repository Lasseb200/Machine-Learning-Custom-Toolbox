function [alpha, b, K] = binary_SVM_training(X_training,Y_training,K_tot,C,idx_pair)
    g = 0;
    K = K_tot(idx_pair,idx_pair);
    H = (Y_training*Y_training').*K;
    %H = H +sqrt(eps)*eye(size(H));
    n = size(X_training,1);
    f = -ones(n,1);
    E = Y_training';
    alpha_min = zeros(n,1); alpha_max = C*ones(n,1);
    opts = optimoptions('quadprog','Display','off','ConstraintTolerance',1e-4,'OptimalityTolerance',1e-4);
    alpha = quadprog(H,f,[],[],E,g,alpha_min,alpha_max,[],opts);
    dynamic_thresh = max(alpha) * 0.001;
    sv_idx = find(alpha > dynamic_thresh & alpha < (C-dynamic_thresh));
    b = mean(Y_training(sv_idx) - ((alpha .* Y_training)' * K(:, sv_idx))');
end
   