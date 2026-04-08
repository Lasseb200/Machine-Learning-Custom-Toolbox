function Y_pred = kmSort(Y_true,Y_pred_idx,costUnmatched)
    Y_true_idx = double(categorical(Y_true));
    Yu = unique(Y_true);
    k_true = length(Yu);
    k_pred = length(unique(Y_pred_idx));
    cost = zeros(k_true, k_pred);
    for i = 1:k_true
        for j = 1:k_pred
            cost(i,j) = -sum((Y_pred_idx==j) & (Y_true_idx==i));
        end
    end
    pairs = matchpairs(cost,costUnmatched);  
    Y_pred_sorted_idx = zeros(size(Y_pred_idx));
    for p = 1:size(pairs,1)
        Y_pred_sorted_idx(Y_pred_idx==pairs(p,2)) = pairs(p,1);
    end
    Y_pred = Yu(Y_pred_sorted_idx);
end

