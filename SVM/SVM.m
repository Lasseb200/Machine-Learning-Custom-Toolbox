function [Y_pred, confidence, probabilities, class_labels] = SVM(X_training,Y_training,X_test,C,kernelType,kernelInput,multiclassModel,executionMode)
    arguments
        X_training
        Y_training
        X_test
        C
        kernelType (1,1) string {mustBeMember(kernelType, ["poly","gaussian","sigmoid"])}
        kernelInput
        multiclassModel (1,1) string {mustBeMember(multiclassModel, ["1vR","hard1v1","soft1v1","pairwiseCoupling"])}
        executionMode (1,1) string {mustBeMember(executionMode, ["single","parallel"])}
    end
    if strcmpi(executionMode,'parallel')
        if isempty(gcp('nocreate'))
            parpool('Processes');
        end
    end
    Ycat = categorical(Y_training);
    Yu = categories(Ycat);
    R = string(Y_training)==string(Yu)';

    kernel = kernel_types(kernelType,kernelInput);
    K_tot = kernel(X_training,X_training);
    [alpha, b, y_allSet_cell, plattA, plattB] = SVM_training(X_training,R,C,K_tot,multiclassModel,executionMode);
    K_test_tot = kernel(X_training,X_test);
    [Y_pred, confidence, probabilities] = SVM_classifier(X_test,alpha,b,K_test_tot,Yu,R,y_allSet_cell,plattA,plattB,multiclassModel);
    class_labels = string(Yu);
end