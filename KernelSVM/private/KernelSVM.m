function [Y_pred, confidence] = KernelSVM(X_training,Y_training,X_test,type,C,input,confidenceModel)
    if strcmpi(confidenceModel, 'none')
        [alpha, b, kernel, Yu, Xr_cell, yr_cell] = KernelSVM_training(X_training,Y_training,type,C,input);
        Y_pred = KernelSVM_1v1classifier(X_test,alpha,b,kernel,Yu,Xr_cell,yr_cell);
        confidence = [];
    elseif strcmpi(confidenceModel, 'ones')
        [alpha, b, kernel, Yu, Xr_cell, yr_cell] = KernelSVM_training(X_training,Y_training,type,C,input);
        [Y_pred, confidence] = KernelSVM_1v1classifier_wconf(X_test,alpha,b,kernel,Yu,Xr_cell,yr_cell);
    elseif strcmpi(confidenceModel, 'Platt')
        [alpha, b, kernel, Yu, Xr_cell, yr_cell,plattA,plattB] = KernelSVM_training_Platt(X_training,Y_training,type,C,input);
        [Y_pred, confidence] = KernelSVM_1v1classifier_wconf_Platt(X_test,alpha,b,kernel,Yu,Xr_cell,yr_cell,plattA,plattB);
    else
        disp("Error: Choose confidence model 'none', 'ones' or 'Platt'")
    end
end