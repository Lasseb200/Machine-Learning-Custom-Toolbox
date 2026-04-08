function [Y_pred, confidence] = SVM_1v1_Classifier_WuLinGPT(X_test, alpha, b, kernel, Yu, Xr_cell, yr_cell, plattA, plattB)

    nPairs = length(b);
    k = (1 + sqrt(1 + 8*nPairs)) / 2;       % number of classes
    pairs = nchoosek(1:k,2);                % class pairs
    n_test = size(X_test,1);
    
    % Preallocate pairwise probability array
    r_ij = zeros(n_test, k, k);  % r_ij(t,i,j) = P(i|i,j)
    
    % Compute pairwise probabilities for each test point
    for p = 1:nPairs
        K_test = kernel(Xr_cell{p}, X_test); 
        F = (alpha{p} .* yr_cell{p})' * K_test - b(p); 
        
        n1 = pairs(p,1); 
        n2 = pairs(p,2);
        
        P_pos = 1 ./ (1 + exp(plattA(p)*F + plattB(p)));  % P(i|i,j)
        P_neg = 1 - P_pos;                                 % P(j|i,j)
        
        r_ij(:,n1,n2) = P_pos;
        r_ij(:,n2,n1) = P_neg;
    end
    
    % Prepare outputs
    %Y_pred = zeros(n_test,1);
    confidence = zeros(n_test,1);
    
    % fmincon options
    opts = optimoptions('fmincon','Display','off','Algorithm','sqp');
    
    % Solve Wu-Lin minimization for each test point
    for t = 1:n_test
        % Initial guess
        p0 = ones(k,1)/k;
        % Equality constraint sum(p)=1
        Aeq = ones(1,k); beq = 1;
        % Lower bound p>=0
        lb = zeros(k,1);
        ub = ones(k,1); % optional upper bound
        
        % Objective function
        fun = @(p) sum(sum(triu( (squeeze(r_ij(t,:,:))' .* p - squeeze(r_ij(t,:,:)) .* p').^2 ,1)));
        
        % Solve minimization
        p_opt = fmincon(fun, p0, [], [], Aeq, beq, lb, ub, [], opts);
        
        % Store prediction and confidence
        [~, idx] = max(p_opt);
        Y_pred(t,1) = Yu(idx);
        confidence(t) = p_opt(idx);
    end
    
    % Optional: normalize confidence across all test points
    confidence = confidence / max(confidence);
end
