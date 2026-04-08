function [W, b, Yu] = LinearSVM_1v1_Training(X_training,Y_training,C)
    Yu = unique(Y_training,'stable');
    R = Y_training==Yu';
    d = size(X_training,2); k = length(Yu);
    pairs = nchoosek(1:k,2); nPairs = size(pairs,1);
    g = 0;
    opts = optimoptions('quadprog','Display','off');
    W = zeros(d,nPairs); b = zeros(nPairs,1);
    for p = 1:nPairs
        n1 = pairs(p,1); n2 = pairs(p,2);
        X1 = R(:,n1).*X_training; X2 = R(:,n2).*X_training;
        X1 = X1(any(X1, 2),:); X2 = X2(any(X2, 2),:);
        y1 = ones(size(X1,1),1); y2 = -ones(size(X2,1),1);
        Xr = [X1;X2]; yr = [y1;y2];
        nr = size(Xr,1);
        f = -ones(nr,1);
        E = yr';
        H = (yr*yr').*(Xr*Xr');
        alpha_min = zeros(nr,1); alpha_max = C*ones(nr,1);
        alpha = quadprog(H,f,[],[],E,g,alpha_min,alpha_max,[],opts);
        W(:,p) = sum(alpha.*yr.*Xr,1);
        sv_idx = find(alpha > eps & alpha < C);
        b(p) = mean(yr(sv_idx) - Xr(sv_idx,:)*W(:,p));
    end
end