#0 silent
#1 +debugging
#2 +basic flow information
#3 +major details
#4 +nn
verbosity_level = 0 
#dirs
seed_dir = '2_mat_data/seed/'
moritz_dir = ['2_mat_data/moritz/eeg/', '2_mat_data/moritz/eye']
stanford_dir = ['2_mat_data/stanford/eeg/', '2_mat_data/stanford/eye']

# storing configuration and accuracy information
# general
dataset_list = []
dataset_list.append('dataset')
name_list = []
name_list.append('names')

# multimodal network
epochs_mm_list = []
epochs_mm_list.append('epochs_mm')
hidden_layer_mm_list = []
hidden_layer_mm_list.append('hidden_layer_units_mm')
act_func_mm_list = []
act_func_mm_list.append('act_func_mm')

# classification nn
hidden_layer_num_list = []
hidden_layer_num_list.append('hidden_layer_num')
hidden_layer_units_list = []
hidden_layer_units_list.append('hidden_layer_units')
training_epochs_list = []
training_epochs_list.append('training_epochs')
stdv_list = []
stdv_list.append('stdv')
init_weight_func_hidden_list = []
init_weight_func_hidden_list.append('init_weight_func_hidden')
init_bias_func_hidden_list = []
init_bias_func_hidden_list.append('init_bias_func_hidden')
init_weight_func_output_list = []
init_weight_func_output_list.append('init_weight_func_output')
init_bias_func_output_list = []
init_bias_func_output_list.append('init_bias_func_output')
act_func_hidden_list = []
act_func_hidden_list.append('act_func_hidden')
dropout_keep_rate_list = []
dropout_keep_rate_list.append('dropout_keep_rate')
initial_learning_rate_list = []
initial_learning_rate_list.append('initial_learning_rate')
seed_list = []
seed_list.append('seed')

#accuracies
svm_eeg_list = []
svm_eeg_list.append('svm_eeg')
svm_eye_list = []
svm_eye_list.append('svm_eye')
svm_mm_list = []
svm_mm_list.append('svm_mm')
acc_nn_list = []
acc_nn_list.append('acc_nn')

suitable_params = {
"adrian_20160411":([50,100,"sigmoid"],[2,500,125,"random_normal","constant","random_normal","truncated_normal","relu"]),
"charlie_20161014":([50,400,"sigmoid"],[2,1100,125,"random_normal","constant","random_normal","constant","sigmoid"]),
"charlie_20161020":([50,60,"sigmoid"],[2,1100,400,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"charlie_20161025":([15,100,"sigmoid"],[2,500,400,"random_normal","constant","random_normal","truncated_normal","tanh"]),
"christoph_20161025":([100,400,"sigmoid"],[2,1100,75,"random_normal","constant","random_normal","constant","sigmoid"]),
"christoph_20161027":([100,140,"sigmoid"],[2,1100,75,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"christoph_20161103":([100,60,"sigmoid"],[2,700,400,"random_normal","constant","random_normal","constant","sigmoid"]),
"dujingcheng_20131027":([15,60,"sigmoid"],[2,500,125,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"dujingcheng_20131030":([15,60,"sigmoid"],[2,500,125,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"dujingcheng_20131107":([15,60,"sigmoid"],[2,500,125,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"jannik_20161021":([100,60,"sigmoid"],[3,700,125,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"jannik_20161111":([100,400,"sigmoid"],[2,500,400,"random_normal","constant","random_normal","truncated_normal","tanh"]),
"jannik_20161118":([100,60,"sigmoid"],[2,500,75,"random_normal","constant","random_normal","constant","sigmoid"]),
"jianglin_20140404":([15,100,"sigmoid"],[3,700,400,"random_normal","constant","random_normal","constant","tanh"]),
"jianglin_20140413":([50,140,"sigmoid"],[2,500,75,"random_normal","constant","random_normal","constant","tanh"]),
"jianglin_20140419":([15,400,"sigmoid"],[2,700,125,"random_normal","constant","random_normal","constant","sigmoid"]),
"jingjing_20140603":([20,400,"sigmoid"],[3,700,400,"random_normal","constant","random_normal","constant","relu"]),
"jingjing_20140610":([100,400,"sigmoid"],[3,700,400,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"jingjing_20140629":([50,60,"sigmoid"],[3,500,125,"random_normal","constant","random_normal","constant","tanh"]),
"liuqiujun_20140621":([50,400,"sigmoid"],[3,700,400,"random_normal","constant","random_normal","constant","relu"]),
"liuqiujun_20140702":([50,60,"sigmoid"],[3,700,125,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"liuqiujun_20140705":([15,400,"sigmoid"],[3,500,125,"random_normal","constant","random_normal","constant","sigmoid"]),
"liuye_20140411":([100,400,"sigmoid"],[3,1100,400,"random_normal","constant","random_normal","constant","relu"]),
"liuye_20140418":([100,400,"sigmoid"],[3,500,75,"random_normal","constant","random_normal","constant","tanh"]),
"liuye_20140506":([50,60,"sigmoid"],[3,700,125,"random_normal","constant","random_normal","truncated_normal","tanh"]),
"martin_20160317":([100,100,"sigmoid"],[2,500,125,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"martin_20160407":([20,60,"sigmoid"],[2,500,75,"random_normal","constant","random_normal","truncated_normal","tanh"]),
"martin2_20160321":([100,140,"sigmoid"],[3,500,400,"random_normal","constant","random_normal","constant","sigmoid"]),
"martin2_20160328":([100,100,"sigmoid"],[3,700,125,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"maya_20160418":([100,100,"sigmoid"],[3,1100,75,"random_normal","constant","random_normal","truncated_normal","relu"]),
"maya_20160425":([100,140,"sigmoid"],[3,700,75,"random_normal","constant","random_normal","constant","tanh"]),
"moritz_20160317":([100,60,"sigmoid"],[2,500,125,"random_normal","constant","random_normal","truncated_normal","tanh"]),
"moritz_20160427":([50,140,"sigmoid"],[3,500,400,"random_normal","constant","random_normal","truncated_normal","tanh"]),
"sunxiangyu_20140511":([100,140,"sigmoid"],[2,500,125,"random_normal","constant","random_normal","truncated_normal","tanh"]),
"sunxiangyu_20140514":([100,60,"sigmoid"],[2,500,125,"random_normal","constant","random_normal","constant","sigmoid"]),
"sunxiangyu_20140521":([50,400,"sigmoid"],[3,500,400,"random_normal","constant","random_normal","truncated_normal","tanh"]),
"wangkui_20140620":([15,100,"sigmoid"],[3,1100,400,"random_normal","constant","random_normal","constant","tanh"]),
"wangkui_20140627":([15,100,"sigmoid"],[3,500,400,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"wangkui_20140704":([100,140,"sigmoid"],[3,700,75,"random_normal","constant","random_normal","truncated_normal","tanh"]),
"weiwei_20131130":([20,100,"sigmoid"],[3,700,400,"random_normal","constant","random_normal","truncated_normal","relu"]),
"weiwei_20131204":([20,100,"sigmoid"],[2,700,400,"random_normal","constant","random_normal","constant","tanh"]),
"weiwei_20131211":([20,400,"sigmoid"],[2,500,400,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"wuyangwei_20131127":([100,140,"sigmoid"],[3,1100,75,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"wuyangwei_20131201":([100,60,"sigmoid"],[3,1100,125,"random_normal","constant","random_normal","truncated_normal","sigmoid"]),
"wuyangwei_20131207":([100,60,"sigmoid"],[3,700,75,"random_normal","constant","random_normal","constant","sigmoid"])}
#'AB0624_20161208':50, 
# 'AB0624_20161216':50,
# 'AL0323_20161206':50,
# 'AL1213_20161223':50,
# 'AL1213_20161227':50,
# 'AR0330_20161206':50,
# 'EL0530_20161219':50,
# 'EL0606_20161220':50,
# 'EL0606_20161227':50,
# 'EL0994_20161220':50,
# 'EL1218_20161207':50,
# 'IG1146_20161205':50,
# 'IG1146_20161208':50,
# 'LH1013_20161226':50,
# 'TI0281_20170103':50}

