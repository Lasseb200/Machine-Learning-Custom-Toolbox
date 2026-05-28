function [Y_pred, confidence,probabilities] = SVM_classifier(X_test,alpha,b,K_test_tot,Yu,R,y_allSet_cell,A_Platt,B_Platt,multiclassModel)
    n_test = size(X_test,1);

    % 1vR
    if strcmpi(multiclassModel, '1vR')
        k = length(Yu);
        F = zeros(n_test,k);
        for j = 1:k
            %K_test = kernel(X_all_cell{j},X_test);
            F(:,j) = (alpha{j}.*y_allSet_cell{j})'*K_test_tot + b(j);
            
        end
        [~, idx_pred] = max(F,[],2);
        Y_pred = string(Yu(idx_pred));
        confidence = NaN;
        probabilities = NaN;
    end

    
    % 1v1 / Pairwise
    if strcmpi(multiclassModel, 'hard1v1') | strcmpi(multiclassModel, 'soft1v1') | strcmpi(multiclassModel, 'pairwiseCoupling')
        nPairs = length(b);
        k = (1+sqrt(1+8*nPairs))/2;
        pairs = nchoosek(1:k,2);
        F = zeros(n_test,nPairs);
        votes = zeros(size(X_test,1),k);

        P = zeros(n_test, k, k);
        for p = 1:nPairs
            j = pairs(p,1); l = pairs(p,2);
            idx_j = R(:,j) == 1;
            idx_l = R(:,l) == 1;
            idx_pair = idx_j | idx_l;
            K_test = K_test_tot(idx_pair,:);
            F(:,p) = (alpha{p}.*y_allSet_cell{p})'*K_test + b(p);

            if strcmpi(multiclassModel, 'hard1v1')
                votes(:,j) = votes(:,j) + (F(:,p)>0);
                votes(:,l) = votes(:,l) + (F(:,p)<0);

            elseif strcmpi(multiclassModel, 'soft1v1')
                P_pos = 1 ./ (1 + exp(A_Platt(p)*F(:,p) + B_Platt(p)));
                P_neg = 1 - P_pos;
                votes(:,j) = votes(:,j) + P_pos;
                votes(:,l) = votes(:,l) + P_neg;

            elseif strcmpi(multiclassModel, 'pairwiseCoupling')
                P_pos = 1 ./ (1 + exp(A_Platt(p)*F(:,p) + B_Platt(p)));
                P(:,j,l) = P_pos;
                P(:,l,j) = 1 - P(:,j,l);
            end
        end

        if strcmpi(multiclassModel, 'hard1v1') | strcmpi(multiclassModel, 'soft1v1')
            [~, idx_pred] = max(votes, [], 2);
            Y_pred = string(Yu(idx_pred));
            totalVotes = sum(votes, 2);
            confidence = votes(sub2ind(size(votes), (1:n_test)', idx_pred)) ./ totalVotes;
            confidence = confidence/max(confidence);
            probabilities = NaN;

        elseif strcmpi(multiclassModel, 'pairwiseCoupling') 
            confidence = zeros(n_test,1);
            Y_pred = strings(n_test,1);
            opts = optimoptions('fmincon','Display','off','Algorithm','sqp');
            for i_test = 1:n_test
                R = squeeze(P(i_test,:,:));
                p0 = ones(k,1)/k;
                Aeq = ones(1,k); beq = 1;
                lb = zeros(k,1); ub = ones(k,1);
                fun = @(p) sum(sum(triu( (R.*repmat(p,1,k)' - R'.*repmat(p,1,k)).^2 ),2),1);
                p_opt = fmincon(fun, p0, [], [], Aeq, beq, lb, ub, [], opts);
                probabilities(i_test,:) = p_opt';
                [~, idx_WL] = max(p_opt);
                confidence(i_test,1) = p_opt(idx_WL);
                Y_pred(i_test) = string(Yu(idx_WL));
            end
        end
    end
end