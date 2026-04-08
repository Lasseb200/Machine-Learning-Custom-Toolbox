function [alpha, b, y_allSet_cell, A_Platt, B_Platt] = SVM_training(X_training,R,C,K_tot,multiclassModel,executionMode)
    k = size(R,2);
    if strcmpi(multiclassModel, '1vR')
        alpha = cell(k,1); y_allSet_cell = cell(k,1); b = zeros(k,1);
        if strcmpi(executionMode, 'parallel')
            parfor j = 1:k
                [alpha{j}, b(j), ~, y_allSet_cell{j}] = multiclass_training_1vR(X_training,R,K_tot,C,j);
            end
        elseif strcmpi(executionMode, 'single')
            for j = 1:k
                [alpha{j}, b(j), ~, y_allSet_cell{j}] = multiclass_training_1vR(X_training,R,K_tot,C,j);
            end
        end
        A_Platt = NaN; B_Platt = NaN;
    end

    if strcmpi(multiclassModel, 'hard1v1') | strcmpi(multiclassModel, 'soft1v1') | strcmpi(multiclassModel, 'pairwiseCoupling')
        pairs = nchoosek(1:k,2); nPairs = size(pairs,1);
        alpha = cell(nPairs,1); y_allSet_cell = cell(nPairs,1); b = zeros(nPairs,1);
        A_Platt = zeros(nPairs,1); B_Platt = zeros(nPairs,1);
        if strcmpi(executionMode, 'parallel')
            parfor p = 1:nPairs
                [alpha{p}, b(p), ~, y_allSet_cell{p}, A_Platt(p), B_Platt(p)] = multiclass_training_pairwise(X_training,R,K_tot,C,pairs,p,multiclassModel);
            end
        elseif strcmpi(executionMode, 'single')
            for p = 1:nPairs
                [alpha{p}, b(p), ~, y_allSet_cell{p}, A_Platt(p), B_Platt(p)] = multiclass_training_pairwise(X_training,R,K_tot,C,pairs,p,multiclassModel);
            end
        end
    end
end