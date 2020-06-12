from adaboost import adaboost_trian,adaClassify
from load_data import read_data
if __name__ =='__main__':
    data,label = read_data()
    classifier_array = adaboost_trian(data,label,9)
    # print(classifier_array)
    re = adaClassify([[5,5],[0,0]],classifier_array)
    print(re)