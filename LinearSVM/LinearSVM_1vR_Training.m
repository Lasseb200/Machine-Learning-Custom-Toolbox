function [W, b, Yu] = LinearSVM_1vR_Training(X_training,Y_training,C)
    Yu = unique(Y_training,'stable');
    YY_training = 2*(Y_training==Yu')-1;
    n = size(X_training,1); d = size(X_training,2); k = size(Yu,1);
    g = 0; f = -ones(n,1);
    alpha_min = zeros(n,1); alpha_max = C*ones(n,1);
    opts = optimoptions('quadprog','Display','off');
    W = zeros(d,k); b = zeros(k,1);
    for j = 1:k
        y = YY_training(:,j);
        H = (y*y').*(X_training*X_training');
        E = y';
        alpha = quadprog(H,f,[],[],E,g,alpha_min,alpha_max,[],opts);
        W(:,j) = sum(alpha.*y.*X_training);
        sv_idx = find(alpha > eps & alpha < C);
        b(j) = mean(y(sv_idx) - X_training(sv_idx,:)*W(:,j));
    end
end