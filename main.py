import multiprocessing
from graph_create import *
from proNE import *
from conv import *
from treelstm_detect import *
from scipy.spatial.distance import cosine
import json
from pathlib import Path




func_dict = graph_create('/root/funcs/', '/root/seed/tp/')
if os.path.exists('/root/phase1/graph/'):
    pass
else:
    os.mkdir('/root/phase1/graph/')
print_graph(func_dict, '/root/phase1/graph/')
graph_vec('/root/phase1/graph/' + 'graph.txt', '/root/phase1/graph/')
print('graph embedding starts')
matrix = runproNE('/root/phase1/graph/', 128, 10, 0.2, 0.5)

def pair_find(target_func, train_feature,target_name):
    phase1_seed = None
    cosine_similarity_values = [target_func.dot(line) / (np.linalg.norm(target_func) * np.linalg.norm(line)) for line in train_feature]
    if max(cosine_similarity_values) > 0.7:
        phase1_seed = reverse_node.get(train_index[np.argmax(cosine_similarity_values)])
    return phase1_seed, target_name

matrix = np.loadtxt('/root/phase1/graph/embeddings_matrix.txt')
train_index = []
nodeset = {}
pred_label = {}
with open('/root/phase1/graph/nodeset.txt', 'r') as f:
    lines = f.readlines()
    for row in lines:
        nodeset[row.split(' ')[1].strip()] = row.split(' ')[0].strip()
reverse_node = {value: key for key, value in nodeset.items()}
for file in os.listdir('/root/seed/tp/'):
    train_file_name = file.split('.')[0].strip()
    train_index.append(nodeset[train_file_name])
train_feature = matrix.take([key for key in train_index], axis = 0)
#read target_set
target_index = []
for file in os.listdir('/root/funcs/'):
    target_file_name = file.split('.')[0].strip()
    try:
        target_index.append(nodeset[target_file_name])
    except KeyError:
        pred_label[target_file_name] = 0
target_feature = matrix.take([key for key in target_index], axis = 0)
best_pair = {}
result_sum = []
process_pool = multiprocessing.Pool(processes=24)
time1 = time.time()
for i in range(len(target_feature)):
    result = process_pool.apply_async(pair_find,args=(target_feature[i],train_feature,reverse_node.get(target_index[i])))
    result_sum.append(result)
process_pool.close()
process_pool.join()
for line in result_sum:
    phase1_seed, phase1_target = line.get()
    if phase1_seed:
        best_pair[phase1_target] = phase1_seed
time2 = time.time()
print(time2-time1)
file_path = '/root/phase2/phase1_res.json'
with open(file_path, 'w') as file:
    json.dump(best_pair, file)
print("Dictionary has been saved to", file_path)

validate = json.load(open('/root/phase2/phase1_res.json'))
if os.path.exists('/root/phase2/target_validate_ast/'):
    shutil.rmtree('/root/phase2/target_validate_ast/')
else:
    os.mkdir('/root/phase2/target_validate_ast/')
target_validate = list(validate.keys())


for phase1_name in target_validate:
    shutil.copy('/root/trees/'+phase1_name+'.tree', '/root/phase2/target_validate_ast/'+phase1_name+'.tree')

time3 = time.time()
validate_files = os.listdir('/root/phase2/target_validate_ast/')
process_pool = multiprocessing.Pool(processes=24)
result_phase2 = []

for file in validate_files:
    validate_seed = validate[file.split('.')[0]]+'.tree'
    result = process_pool.apply_async(conv_multi,args=(validate_seed,file))
    result_phase2.append(result)
process_pool.close()
process_pool.join()
time4 = time.time()
hold_value = [0.95]
final_res = {}
for value in hold_value:
    final_res.clear()
    for line in result_phase2:
        target_final, max_sim = line.get()
        if max_sim > value and target_final:
            final_res[target_final] = max_sim
    print(time4-time3)
    file_path = '/root/phase2/final_res_'+str(value)+'.json'
    with open(file_path, 'w') as file:
        json.dump(final_res, file)
    print("Dictionary has been saved to", file_path)
    #
    shutil.rmtree('/root/phase3/result_final_v1/')
    shutil.rmtree('/root/phase3/result_final_check_v1/')
    os.mkdir('/root/phase3/result_final_v1/')
    os.mkdir('/root/phase3/result_final_check_v1/')

    res_path = '/root/phase2/final_res_'+str(value)+'.json'
    result = json.load(open(res_path))
    for res in result.keys():
        res_name = res.split('.')[0] + '.sol'
        shutil.copy('/root/funcs/'+ res_name,'/root/phase3/result_final_v1/'+res_name)
    files = os.listdir('/root/phase3/result_final_v1/')
