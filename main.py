from mldata import *
# from test_mldata import *
from math import log2
import numpy as np
from Naive_Bayes_boosting import *
from scipy.io import arff
# import pandas as pd

def main():
    dataset = parse_c45("volcanoes")
    dataset = ExampleSet(dataset)
    trainset = ExampleSet(ex for ex in dataset if int(ex[0])<2000)
    testset = ExampleSet(ex for ex in dataset if int(ex[0])>2000)
    data_read = trainset
    k = 3
    m = 5
    iteration = 1

    weight = 1/ len(data_read) * np.ones(len(data_read))
    weight = weight.reshape(-1,1)
    error_ind = []


    save_prediction_label = np.zeros([len(data_read),1])
    save_test_label = np.zeros([len(testset),1])
    save_target_label_list = [x[-1] for x in data_read]
    save_target_label = np.array(save_target_label_list)
    save_target_label = save_target_label.reshape(-1,1)
    save_alpha = []

    for iiter in range(iteration):    
        data_newarray = np.array(data_read.to_float())
        new_data = data_newarray[:, 1:-1]
        maxminarray = np.zeros([2,len(new_data[0])])
        for i in range(1, len(data_read[0])-1):
            if data_read.schema.features[i].type == 'CONTINUOUS':
                maxminarray[0, i-1] = np.min(new_data[:,i-1])
                maxminarray[1, i-1] = np.max(new_data[:,i-1])
        labels_ratio, save_all_prob, save_all_threshold = showme_dataset(data_read,k,m,maxminarray,weight,error_ind)

        for j in range(len(data_read)):
            the_example = data_read[j]
            operated_example = operate_example(data_read,the_example,save_all_threshold)
            predict_label = showme_example(operated_example,save_all_prob, labels_ratio)
            save_prediction_label[j] = predict_label
        error_ind, a = np.where(save_prediction_label != save_target_label)
        print("error_ind = ", len(error_ind))
        weight,episilon,alpha = boosting_updata_weight( weight, error_ind ,save_prediction_label, save_target_label )
        save_alpha.append(alpha)
        print('accuracy = ', 1-(len(error_ind)/len(trainset)) )
        print('alpha =', alpha)
        print("episilon = ",episilon,"\n")
        if episilon > 0.5 or episilon == 0 or episilon == 0.5:
            print("episilon>0.5")
            break
  
    save_test_predict_label_temp = [x[-1] for x in testset]
    save_test_predict_label = np.array(save_test_predict_label_temp).reshape(len(save_test_predict_label_temp),1)

    save_test_predict_label_1 = np.where(save_test_predict_label == 1)
    save_test_predict_label_0 = np.where(save_test_predict_label == 0)
    labels_1_values = len(save_test_predict_label_1[0])
    labels_0_values = len(save_test_predict_label_0[0])
    labels_1_ratio = labels_1_values/(labels_1_values+labels_0_values)
    labels_0_ratio = labels_0_values/(labels_1_values+labels_0_values)
    test_label_ratio = [[labels_1_ratio,labels_0_ratio]]
    for j in range(len(testset)): 
        the_example = testset[j]
        operated_example = operate_example(data_read,the_example,save_all_threshold)
        predict_label = showme_example(operated_example,save_all_prob, test_label_ratio)
        save_test_label[j] = predict_label
    
    error_ind, a = np.where(save_test_label != save_test_predict_label)
    print(error_ind)
    print('test accuracy = ', 1-(len(error_ind)/len(testset)) )
    print('test alpha =', alpha,"\n")

    # path = './yeast/yeast-train.arff'
    # data, meta = arff.loadarff(path)
    # df = pd.DataFrame(data)
    # print(df)
    # print(meta)



if __name__ == '__main__':
    main()