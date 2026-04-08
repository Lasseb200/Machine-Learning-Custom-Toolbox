function [Y_pred, acc_kNN, acc_SVM, acc_combined] = SmartKernelSVM_vs_kNN_wacc(amplitude_training,phase_training,Y_training,amplitude_test,phase_test,Y_test)
    [Y_pred_kNN, kNN_Confidence] = kNN_wconf(phase_training,Y_training,phase_test,5);
    acc_kNN = mean(Y_test == Y_pred_kNN)*100;
    highConfidence = abs(kNN_Confidence-1)<1e-6;
    Extra_amplitude_training = amplitude_test(highConfidence,:);
    Extra_Y_training = Y_pred_kNN(highConfidence);
    amplitude_training = [amplitude_training;Extra_amplitude_training];
    Y_training = [Y_training;Extra_Y_training];
    [Y_pred_SVM, SVM_Confidence] = KernelSVM_wconf(amplitude_training,Y_training,amplitude_test,'poly',1,{1,0});
    acc_SVM = mean(Y_test == Y_pred_SVM)*100;
    [~, idx] = max([kNN_Confidence./sum(kNN_Confidence) SVM_Confidence./sum(SVM_Confidence)], [], 2);
    Y_pred_models = [Y_pred_kNN Y_pred_SVM];
    n_test = size(amplitude_test,1);
    Y_pred = Y_pred_models(sub2ind(size(Y_pred_models), (1:n_test)', idx));
    acc_combined = mean(Y_test == Y_pred)*100;
end