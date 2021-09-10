
import numpy as np
import pandas as pd
from Fairness_metrics import Fairness_metrics

def combine_list(List):
    '''Combine the list
    '''
    var = []
    for j in range(0, len(List)):
        var = var + List[j] # notice the order!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return var

def calculate_distance(softmax_output, softmax_output_grad):
    '''
    Calculate the distance between each point to the decision boundary
    '''
    softmax_output = combine_list(softmax_output)
    softmax_output_grad = combine_list(softmax_output_grad)
    distance = []

    for i in range(len(softmax_output)):
        d_up = abs(softmax_output[i][0] - softmax_output[i][1])
        d_down = np.linalg.norm(softmax_output_grad[i][0] - softmax_output_grad[i][1])
        distance.append(d_up / d_down)
    return distance

def select_adversarial_examples(test_set, adversarial_set, distance_list, method, sen_attribute, num):
    '''Select 20% data points to generate adversarial examples

    In group 1, the goal is to find some points that turn predicted labels from '1' to '0'.
    In group 0, the goal is to find some points that turn predicted labels from '0' to '1'.

    '''

    test_set['distance'] = distance_list
    test_set['orig_pred'] = adversarial_set['orig_pred']

    # select data points in two groups according to the sensitive attribute and labels
    test_group_1 = test_set[(test_set[sen_attribute] == 1) & (test_set['orig_pred'] == 1)]
    test_group_0 = test_set[(test_set[sen_attribute] == 0) & (test_set['orig_pred'] == 0)]

    # combine these data points together and sort according to the distance from decision boundary
    test_group = pd.concat([test_group_1, test_group_0])
    test_sorted_distance = test_group.sort_values('distance')

    # select the top 20% data points
    #num = int(0.2 * test_set.shape[0])
    if method == 'adv':
        selected_adv_data = test_sorted_distance[:num]
    else:
        selected_adv_data = test_set.sample(n=num)

    # obtain the adversarial set that combines selected adversarial examples and original test examples
    selected_adv_set = adversarial_set.loc[selected_adv_data.index.tolist()]
    selected_test_set = test_set.loc[selected_adv_data.index.tolist()]

    mixed_adversarial_set = test_set.copy()  # notice
    for i in selected_adv_set.index:
        mixed_adversarial_set['orig_pred'].loc[i] = selected_adv_set['adv_pred'].loc[i] # notice the order
        #mixed_adversarial_set.loc[i] = selected_adv_set.loc[i] # notice the order
    return mixed_adversarial_set, selected_adv_set, selected_test_set

def group_size(data, sen_attribute):
    '''
    '''
    group_1_1 = data[(data[sen_attribute] == 1) & (data['orig_pred'] == 1)]
    group_0_0 = data[(data[sen_attribute] == 0) & (data['orig_pred'] == 0)]
    group_1_0 = data[(data[sen_attribute] == 1) & (data['orig_pred'] == 0)]
    group_0_1 = data[(data[sen_attribute] == 0) & (data['orig_pred'] == 1)]

    return group_1_1.shape, group_0_0.shape, group_1_0.shape, group_0_1.shape

def check_discrimination(mixed_adversary_set, sen_attribute, pred):
    ''' Check the discrimination level on the mixed adversary set

    mixed adversary set: include original data and selected adversary examples

    '''
    mixed_adversary_set = mixed_adversary_set.reset_index(drop=True)
    fairness_metrics = Fairness_metrics(mixed_adversary_set, sen_attribute, pred)
    demographic_parity = fairness_metrics.demographic_parity()
    return demographic_parity