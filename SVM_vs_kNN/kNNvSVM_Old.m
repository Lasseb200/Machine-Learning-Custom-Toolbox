function [Y_pred_combined, Y_pred_kNN, Y_pred_SVM, acc_combined, acc_kNN, acc_SVM, kNN_confidence, SVM_confidence] = kNNvSVM_Old(amplitude_training,phase_training,Y_training,amplitude_test,phase_test,f,kernelType,C,SVM_input,Y_test)
    % kNN model for unwrapped phase
    [Y_pred_kNN, kNN_confidence] = kNN(phase_training,Y_training,phase_test,f);
    acc_kNN = mean(Y_test == Y_pred_kNN)*100;
    % SVM model for amplitude
    [Y_pred_SVM, SVM_confidence] = SVM_Old(amplitude_training,Y_training,amplitude_test,kernelType,C,SVM_input,'Platt');
    acc_SVM = mean(Y_test == Y_pred_SVM)*100;
    % Confidence vote
    [~, idx] = max([kNN_confidence SVM_confidence], [], 2);
    Y_pred_models = [Y_pred_kNN Y_pred_SVM];
    n_test = size(amplitude_test,1);
    Y_pred_combined = Y_pred_models(sub2ind(size(Y_pred_models), (1:n_test)', idx));
    acc_combined = mean(Y_test == Y_pred_combined)*100;
end
