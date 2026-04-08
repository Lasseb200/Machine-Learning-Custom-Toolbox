function [Y_pred_combined, Y_pred_kNN, Y_pred_SVM, acc_combined, acc_kNN, acc_SVM, kNN_confidence, SVM_confidence] = kNNvSVM(amplitude_training,phase_training,Y_training,amplitude_test,phase_test,f,C,kernelType,kernelInput,Y_test,SVM_multiclassModel,executionMode)
    arguments
        amplitude_training
        phase_training
        Y_training
        amplitude_test
        phase_test
        f
        C
        kernelType (1,1) string {mustBeMember(kernelType, ["poly","gaussian","sigmoid"])}
        kernelInput
        Y_test
        SVM_multiclassModel (1,1) string {mustBeMember(SVM_multiclassModel, ["1vR","hard1v1","soft1v1","pairwiseCoupling"])}
        executionMode (1,1) string {mustBeMember(executionMode, ["single","parallel"])}
    end
    % kNN model for unwrapped phase
    [Y_pred_kNN, kNN_confidence] = kNN(phase_training,Y_training,phase_test,f);
    acc_kNN = mean(Y_test == Y_pred_kNN)*100;
    % SVM model for amplitude
    [Y_pred_SVM, SVM_confidence] = SVM(amplitude_training,Y_training,amplitude_test,C,kernelType,kernelInput,SVM_multiclassModel,executionMode);
    acc_SVM = mean(Y_test == Y_pred_SVM)*100;
    % Confidence vote
    [~, idx] = max([kNN_confidence SVM_confidence], [], 2);
    Y_pred_models = [Y_pred_kNN Y_pred_SVM];
    n_test = size(amplitude_test,1);
    Y_pred_combined = Y_pred_models(sub2ind(size(Y_pred_models), (1:n_test)', idx));
    acc_combined = mean(Y_test == Y_pred_combined)*100;
end
