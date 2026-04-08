function [alpha, b, kernel, Yu, Xr_cell, yr_cell, plattA, plattB] = KernelSVM_training_Platt(X_training,Y_training,type,C,input)
    kernel = kernelTypes(type,input);
    Yu = unique(Y_training,'stable');
    R = Y_training==Yu';
    k = size(Yu,1);
    pairs = nchoosek(1:k,2); nPairs = size(pairs,1);
    g = 0;
    alpha = cell(nPairs,1); Xr_cell = cell(nPairs,1); yr_cell = cell(nPairs,1);
    b = zeros(nPairs,1);
    plattA = zeros(nPairs,1);
    plattB = zeros(nPairs,1);
    opts = optimoptions('quadprog','Display','off');
    for p = 1:nPairs
        n1 = pairs(p,1); n2 = pairs(p,2);
        X1 = R(:,n1).*X_training; X2 = R(:,n2).*X_training;
        X1 = X1(any(X1, 2),:); X2 = X2(any(X2, 2),:);
        y1 = ones(size(X1,1),1); y2 = -ones(size(X2,1),1);
        Xr = [X1;X2]; yr = [y1;y2]; Xr_cell{p} = Xr; yr_cell{p} = yr;
        nr = size(Xr,1);
        f = -ones(nr,1);
        E = yr';
        K = kernel(Xr,Xr);
        H = (yr*yr').*K;
        alpha_min = zeros(nr,1); alpha_max = C*ones(nr,1);
        alpha{p} = quadprog(H,f,[],[],E,g,alpha_min,alpha_max,[],opts);
        sv_idx = find(alpha{p} > eps & alpha{p} < C);
        b(p) = mean(yr(sv_idx) - ((alpha{p}.*yr)'*K(:,sv_idx))');


        %%% Platt Parameters %%%
        F_train = ((alpha{p} .* yr)' * K)';  % n_pair x 1
        F_train = F_train + b(p);
        % Fit Platt scaling parameters
        t = (yr + 1)/2;  % convert labels +1/-1 -> 1/0

        negLL = @(ab) -sum(t.*log(1./(1+exp(ab(1)*F_train+ab(2))))+(1-t).*log(1-1./(1+exp(ab(1)*F_train+ab(2)))));
        ab0 = [0 0];
        options = optimoptions('fminunc','Display','off','Algorithm','quasi-newton');
        ab_opt = fminunc(negLL, ab0, options);
        plattA(p) = ab_opt(1);
        plattB(p) = ab_opt(2);
    end
end