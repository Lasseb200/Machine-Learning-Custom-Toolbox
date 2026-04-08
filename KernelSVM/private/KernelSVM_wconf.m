function [Y_pred, confidence] = KernelSVM_wconf(X_training,Y_training,X_test,type,C,input)
    [alpha, b, kernel, Yu, Xr_cell, yr_cell] = KernelSVM_training(X_training,Y_training,type,C,input);
    [Y_pred, confidence] = KernelSVM_1v1classifier_wconf(X_test,alpha,b,kernel,Yu,Xr_cell,yr_cell);
end