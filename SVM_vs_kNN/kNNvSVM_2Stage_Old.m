function [Y_pred_total, acc_total, acc_combined_chem, acc_kNN_chem, acc_SVM_chem, acc_combined_conc, acc_kNN_conc, acc_SVM_conc] = kNNvSVM_2Stage_Old(amplitude_training,phase_training,Y_training,Y_training_chem,amplitude_test,phase_test,Y_test,Y_test_chem,chemicals)
% Stage 1
[Y_pred_chem, ~, ~, acc_combined_chem, acc_kNN_chem, acc_SVM_chem] = kNNvSVM_Old(amplitude_training,phase_training,Y_training_chem,amplitude_test,phase_test,5,'poly',1,{1,0},Y_test_chem);

% Stage 2
uniqueChems = unique(chemicals,'stable');
clear Y_pred_kNN_conc Y_pred_SVM_conc Y_pred_conc
Y_pred_total = repmat("name", length(Y_test), 1);
Y_pred_total_kNN = repmat("name", length(Y_test), 1);
Y_pred_total_SVM = repmat("name", length(Y_test), 1);
for k = 1:length(uniqueChems)
    chem = uniqueChems(k);
    
    %%% Training Data %%%
    idx_conc_training = Y_training_chem == chem; 
    phase_training_conc = phase_training(idx_conc_training,:);
    amplitude_traning_conc = amplitude_training(idx_conc_training,:);
    Y_training_conc = Y_training(idx_conc_training);

    %%% Test Data %%%
    idx_conc_test = Y_pred_chem == chem;
    phase_test_conc = phase_test(idx_conc_test,:);
    amplitude_test_conc = amplitude_test(idx_conc_test,:);
    Y_test_conc = Y_test(idx_conc_test);

    if length(unique(Y_training_conc)) < 2

        Y_pred_conc{k} = Y_test_conc;
        Y_pred_kNN_conc{k} = Y_test_conc; Y_pred_SVM_conc{k} = Y_test_conc;

    else
        [Y_pred_conc{k}, Y_pred_kNN_conc{k}, Y_pred_SVM_conc{k}] = kNNvSVM_Old(amplitude_traning_conc,phase_training_conc,Y_training_conc,amplitude_test_conc,phase_test_conc,5,'poly',1,{1,0},Y_test_conc);
    end
    idx_list = find(idx_conc_test);
    Y_pred_total(idx_list) = Y_pred_conc{k};
    Y_pred_total_kNN(idx_list) = Y_pred_kNN_conc{k};
    Y_pred_total_SVM(idx_list) = Y_pred_SVM_conc{k};
end
acc_kNN_conc = mean(Y_test == Y_pred_total_kNN)/acc_combined_chem*1e4;
acc_SVM_conc = mean(Y_test == Y_pred_total_SVM)/acc_combined_chem*1e4;
acc_combined_conc = mean(Y_test == Y_pred_total)/acc_combined_chem*1e4;
acc_total = mean(Y_test == Y_pred_total)*100;

end