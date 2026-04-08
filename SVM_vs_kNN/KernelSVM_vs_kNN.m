function Y_pred = KernelSVM_vs_kNN(X_training_kNN,Y_training_kNN,X_training_SVM,Y_training_SVM,X_test_kNN,X_test_SVM,f,type,C,input)
    tic
    [Y_pred_kNN, kNN_Confidence] = kNN_wconf(X_training_kNN,Y_training_kNN,X_test_kNN,f);
    elapsedTime = toc;  % stop timer
    fprintf('kNN finished in %.3f seconds.\n', elapsedTime);
    tic
    [Y_pred_SVM, SVM_Confidence] = KernelSVM_wconf(X_training_SVM,Y_training_SVM,X_test_SVM,type,C,input);
    elapsedTime = toc;  % stop timer
    fprintf('SVM finished in %.3f seconds.\n', elapsedTime);
    [~, idx] = max([kNN_Confidence SVM_Confidence], [], 2);
    Y_pred_models = [Y_pred_kNN Y_pred_SVM];
    n_test = size(X_test_SVM,1);
    Y_pred = Y_pred_models(sub2ind(size(Y_pred_models), (1:n_test)', idx));
end