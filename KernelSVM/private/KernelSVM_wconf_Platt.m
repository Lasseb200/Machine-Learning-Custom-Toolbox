function [Y_pred, confidence] = KernelSVM_wconf_Platt(X_training,Y_training,X_test,type,C,input)
    [alpha, b, kernel, Yu, Xr_cell, yr_cell,plattA,plattB] = KernelSVM_training_Platt(X_training,Y_training,type,C,input);
    [Y_pred, confidence] = KernelSVM_1v1classifier_wconf_Platt(X_test,alpha,b,kernel,Yu,Xr_cell,yr_cell,plattA,plattB);
end