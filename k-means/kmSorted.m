function [Y_pred, confidence, M] = kmSorted(X,k,Y_true,costUnmatched)
    [Y_pred_idx, confidence, M] = km(X,k);
    Y_pred = kmSort(Y_true,Y_pred_idx,costUnmatched);
end