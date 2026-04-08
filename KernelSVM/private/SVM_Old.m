function [Y_pred, confidence] = SVM_Old(X_training,Y_training,X_test,kernelType,C,input,confidenceModel)
    [alpha, b, kernel, Yu, Xr_cell, yr_cell,plattA,plattB] = SVM_training(X_training,Y_training,kernelType,C,input,confidenceModel);
    [Y_pred, confidence] = SVM_1v1_Classifier(X_test,alpha,b,kernel,Yu,Xr_cell,yr_cell,plattA,plattB,confidenceModel);
end