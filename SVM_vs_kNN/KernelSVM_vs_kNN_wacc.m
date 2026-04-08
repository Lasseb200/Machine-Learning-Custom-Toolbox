function [Y_pred, acc_kNN, acc_SVM, acc_combined, Y_pred_kNN, Y_pred_SVM, kNN_Confidence, SVM_Confidence] = KernelSVM_vs_kNN_wacc(X_training_kNN,Y_training_kNN,X_training_SVM,Y_training_SVM,X_test_kNN,X_test_SVM,f,type,C,input,Y_test)
    %tic
    [Y_pred_kNN, kNN_Confidence] = kNN_wconf(X_training_kNN,Y_training_kNN,X_test_kNN,f);
    acc_kNN = mean(Y_test == Y_pred_kNN)*100;
    %elapsedTime = toc;  % stop timer
    %fprintf('kNN finished in %.3f seconds with accuracy %.2f%%.\n', elapsedTime, acc);
    %tic
    [Y_pred_SVM, SVM_Confidence] = KernelSVM_wconf(X_training_SVM,Y_training_SVM,X_test_SVM,type,C,input);
    acc_SVM = mean(Y_test == Y_pred_SVM)*100;
    %elapsedTime = toc;  % stop timer
    %fprintf('SVM finished in %.3f seconds with accuracy %.2f%%.\n', elapsedTime, acc);
    [~, idx] = max([kNN_Confidence./sum(kNN_Confidence) SVM_Confidence./sum(SVM_Confidence)], [], 2);
    Y_pred_models = [Y_pred_kNN Y_pred_SVM];
    n_test = size(X_test_SVM,1);
    Y_pred = Y_pred_models(sub2ind(size(Y_pred_models), (1:n_test)', idx));
    %tol = 1e-6;
    %use_knn = abs(kNN_Confidence - 1) < tol;
    %Y_pred = Y_pred_SVM;     % start with SVM everywhere
    %Y_pred(use_knn) = Y_pred_kNN(use_knn);
    acc_combined = mean(Y_test == Y_pred)*100;
end
