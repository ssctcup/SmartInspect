import ctypes.wintypes
import os
import subprocess

import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance
import time
import shutil
from Tree import Tree

# target_sol_path = 'target_validate/'
# seed_sol_path = 'seed_sol/'

# target_ast_path = 'target_validate_ast/'
# seed_ast_path = 'seed_ast/'
# to_next_ast_path = 'target_next_validate/'
# hold_value = 0.75

vocab_list = []
target_tree_files = []
prob_seed_files = []
to_next_tree_files = []

device = torch.device("cuda")




def make_tree_structure(tree_str, up_node, layer_flag):
    tmp_str = tree_str
    new_node = Tree()
    if len(tmp_str.split(" ")) == 1:
        new_node.opname = tmp_str.strip('()')
        up_node.add_child(new_node)
        return
    elif len(tmp_str.split()) == 0:
        return
    else:  #
        add_par_flag = 0
        if list(tmp_str)[0] + list(tmp_str)[1] == '( ':
            new_lpar_node = Tree()
            new_lpar_node.opname = '('
            up_node.add_child(new_lpar_node)
            add_par_flag = 1
        tmp_str = tmp_str.strip().strip('()').strip()
        new_node.opname = tmp_str.split(" ")[0].strip('()')
        up_node.add_child(new_node)
        split_list = []
        parentheses_flag = 0
        sub_str = ''
        for ch in tmp_str:
            if parentheses_flag == 0 and ch != ' ' and ch != '(':
                sub_str += ch
            elif parentheses_flag == 0 and ch == ' ':
                split_list.append(sub_str)
                sub_str = ''
            elif parentheses_flag == 0 and ch == '(':
                sub_str += ch
                parentheses_flag = 1
            elif parentheses_flag != 0 and ch == '(':
                sub_str += ch
                parentheses_flag += 1
            elif parentheses_flag != 0 and ch == ')':
                sub_str += ch
                parentheses_flag -= 1
            else:
                sub_str += ch
        split_list.append(sub_str)
        sub_str = ''

        for i in range(1 - add_par_flag, len(split_list)):  # [word, (), (), ()]
            # print(split_list[i])  # ( (word () () ()) )|word () () ()
            make_tree_structure(split_list[i], new_node, 0)

        if add_par_flag:
            new_rpar_node = Tree()
            new_rpar_node.opname = ')'
            up_node.add_child(new_rpar_node)
        return


def get_vocab(node_list):
    global vocab_list
    # vocab_list = []
    for i in range(len(node_list)):
        if node_list[i] not in vocab_list:
            vocab_list.append(node_list[i])


def preorder_visit_tree(tree_node, node_list, index, index_list):
    node_list.append(tree_node.opname)
    index_list.append(index)
    tree_node.idx = index
    for i in range(tree_node.num_children):
        index += 1
        tmp, index, index_list = preorder_visit_tree(tree_node.children[i], node_list, index, index_list)
    return node_list, index, index_list


def getposition(node_name, name_list, node_index):  # name_list
    for i in range(len(name_list)):
        if name_list[i] is node_name and i != node_index:
            mm = 0  # opname
        if name_list[i] is node_name and i == node_index:
            return i
    return 0


def construct_adj_matrix(root, matrix, name_list):
    i = getposition(root.opname, name_list, root.idx)
    # print(root.opname,len(root.opname))
    for j in range(root.num_children):
        k = getposition(root.children[j].opname, name_list, root.children[j].idx)
        # print(i, k)
        matrix[i][k] = 1
        matrix[k][i] = 1
        construct_adj_matrix(root.children[j], matrix, name_list)
    return matrix

def get_adj(file):
    tree_string = open(file, 'r').read()
    tmp_node_list = []
    tmp_index = 0
    tmp_root = Tree()
    make_tree_structure(tree_string, tmp_root, 0)
    tmp_root = tmp_root.children[0]
    tmp_root.value = file
    tmp_index_list = []
    preorder_visit_tree(tmp_root, tmp_node_list, tmp_index, tmp_index_list)
    # tmp_adj_matrix = [[0 for i in range(len(tmp_node_list))] for j in
    #                   range(len(tmp_node_list))]
    tmp_adj_matrix = np.zeros((len(tmp_node_list),len(tmp_node_list)))
    tmp_adj_matrix = construct_adj_matrix(tmp_root, tmp_adj_matrix, tmp_node_list)
    return tmp_adj_matrix
    # print(tmp_index_list)
    # get_vocab(tmp_node_list)


def get_adj_matrix(filepath, cnt):
    trees = []
    tree_string_lists = []
    tree_node_lists = []
    matrix_list = []

    files = os.listdir(filepath)
    for file in files:
        tree_string = open(filepath + file, 'r').read()
        tree_string_lists.append(tree_string)
        tmp_node_list = []
        tmp_index = 0
        tmp_root = Tree()
        make_tree_structure(tree_string, tmp_root, 0)

        tmp_root = tmp_root.children[0]

        print(file)
        tmp_root.value = file
        tmp_index_list = []
        preorder_visit_tree(tmp_root, tmp_node_list, tmp_index, tmp_index_list)
        # print(tmp_index_list)
        get_vocab(tmp_node_list)
        trees.append(tmp_root)
        tree_node_lists.append(tmp_node_list)
    for i in range(len(trees)):
        if len(tree_node_lists[i]) > cnt:
            tmp_adj_matrix = [[0 for i in range(len(tree_node_lists[i]))] for j in
                                  range(len(tree_node_lists[i]))]
            # print(len(tmp_adj_matrix), len(tmp_adj_matrix[0]))
            construct_adj_matrix(trees[i], tmp_adj_matrix, tree_node_lists[i])
            matrix_list.append(tmp_adj_matrix)
            if cnt ==1:
                target_tree_files.append(trees[i].value)
    return matrix_list


def th_gather_nd(x, coords):
    x = x.contiguous()
    inds = coords.mv(torch.IntTensor(x.stride()))
    x_gather = torch.index_select(x.contiguous().view(-1), 0, inds)
    return x_gather


# def run():
#     time1 = time.perf_counter()  #########
#     ###
#     seed_adj_matrix = get_adj_matrix(seed_ast_path)
#     target_adj_matrix = get_adj_matrix(target_ast_path, len(seed_adj_matrix[0]))
#     time2 = time.perf_counter()  #########
#     print(target_tree_files)
#     print(prob_seed_files)
#     # N = (W ? F + 2P ) / S + 1
#     #             torch.tensor
#     ###
#     seed_adj_np_matrix = np.array([[seed_adj_matrix[0]]])
#     seed_tensor = torch.from_numpy(seed_adj_np_matrix)
#     seed_tensor = seed_tensor.double()
#     seed_tensor = seed_tensor.to(device)
#     print(seed_tensor)
#     ###
#     l_seed = len(seed_adj_matrix[0])
#     c = torch.nn.Conv2d(1, 1, kernel_size=l_seed, stride=1, padding=0, bias=False)
#     c.weight.data = seed_tensor
#     c = c.to(device)
#     print(len(seed_adj_matrix[0]))
#     max_nums = []
#     time_step1 = time2 - time1  #########
#     time1 = time.perf_counter()
#     time_conv = 0
#     ###
#     for k in range(len(target_tree_files)):
#         print('start' + target_tree_files[k])
#
#         try:
#             target_adj_np_matrix = np.array([[target_adj_matrix[k]]])
#             target_tensor = torch.from_numpy(target_adj_np_matrix)
#             target_tensor = target_tensor.double()
#             target_tensor = target_tensor.to(device)
#             # print(target_tensor)
#             time_begin = time.perf_counter()
#             result = c(target_tensor)
#             time_end = time.perf_counter()
#             time_conv += (time_end - time_begin)
#             result = result.squeeze()
#             print(result)
#             tmp = torch.where(torch.gt(result, 0) & torch.lt(result, hold_value * (l_seed - 1)),
#                               torch.zeros_like(result), result)
#             # print(torch.nonzero(tmp))
#             idxs = torch.where(result > hold_value)
#             idxs = torch.nonzero(tmp)
#             print('--V idxs---')
#             print(idxs)
#             tmp_max_nums = []
#             if idxs.numel():
#                 print(idxs.shape[0], idxs.shape[1])
#                 for i in range(idxs.shape[0]):
#                     tmp_max_nums.append(tmp[idxs[i][0].item()][idxs[i][1].item()].item() / (l_seed - 1))
#                 # print(tmp_max_nums)
#
#             # maxnum = torch.max(result)
#             # final_maxnum = maxnum.item() / (l_seed - 1)  # 0
#             # max_nums.append(final_maxnum)
#             max_nums.append(tmp_max_nums)
#             if len(tmp_max_nums) > 0:
#                 to_next_tree_files.append(target_tree_files[k])
#             print('--V max---')
#             print(tmp_max_nums)
#         except:
#             raise
#     time2 = time.perf_counter()
#     time_step2 = time2 - time1
#     print('to csv')
#     print('num of files :', len(target_tree_files))
#     print('to next files :', len(to_next_tree_files))
#     for i in range(len(to_next_tree_files)):
#         shutil.copyfile(target_ast_path + to_next_tree_files[i], to_next_ast_path + to_next_tree_files[i])
#     print('copy file done')
#     data = pd.DataFrame({'target': target_tree_files, 'result': max_nums})
#     data.to_csv('result/vul-5629-0-1-0-1-org-0.75.csv')  # 0
#     print('step_1: ', time_step1)
#     print('step_2: ', time_step2)
#     return time_step1, time_step2, time_conv


# start = time.perf_counter()
# with torch.no_grad():
#     time_1, time_2, time_conv = run()
# end = time.perf_counter()
# time_total = end - start
# print(time_total)
# times = [time_1, time_2, time_conv, time_total]
# timedata = pd.DataFrame({'Time': times})
# timedata.to_csv('result/gpu_01_vul_pure_conv_time_simp_0.75.csv')


# def conv_process(seed_ast_path, target_ast_path, to_next_ast_path, hold_value):
#     time1 = time.perf_counter()  #########
#     ###
#     seed_adj_matrix = get_adj_matrix(seed_ast_path)
#     target_adj_matrix = get_adj_matrix(target_ast_path, len(seed_adj_matrix[0]))
#     time2 = time.perf_counter()  #########
#     print(target_tree_files)
#     print(prob_seed_files)
#     seed_adj_np_matrix = np.array([[seed_adj_matrix[0]]])
#     seed_tensor = torch.from_numpy(seed_adj_np_matrix)
#     seed_tensor = seed_tensor.double()
#     seed_tensor = seed_tensor.to(device)
#     print(seed_tensor)
#     ###
#     l_seed = len(seed_adj_matrix[0])
#     c = torch.nn.Conv2d(1, 1, kernel_size=l_seed, stride=1, padding=0, bias=False)
#     c.weight.data = seed_tensor
#     c = c.to(device)
#     print(len(seed_adj_matrix[0]))
#     max_nums = []
#     time_step1 = time2 - time1  #########
#     time1 = time.perf_counter()
#     time_conv = 0
#     ###
#     for k in range(len(target_tree_files)):
#         print('start' + target_tree_files[k])
#
#         try:
#             target_adj_np_matrix = np.array([[target_adj_matrix[k]]])
#             target_tensor = torch.from_numpy(target_adj_np_matrix)
#             target_tensor = target_tensor.double()
#             target_tensor = target_tensor.to(device)
#             # print(target_tensor)
#             time_begin = time.perf_counter()
#             result = c(target_tensor)
#             time_end = time.perf_counter()
#             time_conv += (time_end - time_begin)
#             result = result.squeeze()
#             print(result)
#             tmp = torch.where(torch.gt(result, 0) & torch.lt(result, hold_value * (l_seed - 1)),
#                               torch.zeros_like(result), result)
#             # print(torch.nonzero(tmp))
#             # idxs = torch.where(result > hold_value)
#             idxs = torch.nonzero(tmp)
#             print('--Max sim pos---')
#             print(idxs)
#             tmp_max_nums = []
#             if idxs.numel():
#                 # print(idxs.shape[0], idxs.shape[1])
#                 for i in range(idxs.shape[0]):
#                     tmp_max_nums.append(tmp[idxs[i][0].item()][idxs[i][1].item()].item() / (l_seed - 1))
#                 # print(tmp_max_nums)
#
#             # maxnum = torch.max(result)
#             # final_maxnum = maxnum.item() / (l_seed - 1)  # 0
#             # max_nums.append(final_maxnum)
#             max_nums.append(tmp_max_nums)
#             if len(tmp_max_nums) > 0:
#                 to_next_tree_files.append(target_tree_files[k])
#             print('--Sim max---')
#             print(tmp_max_nums)
#         except:
#             raise
#     time2 = time.perf_counter()
#     time_step2 = time2 - time1
#     print('to csv')
#     print('num of files :', len(target_tree_files))
#     print('to next files :', len(to_next_tree_files))
#     if os.path.exists(to_next_ast_path):
#         shutil.rmtree(to_next_ast_path)
#     os.mkdir(to_next_ast_path)
#     for i in range(len(to_next_tree_files)):
#         shutil.copyfile(target_ast_path + to_next_tree_files[i], to_next_ast_path + to_next_tree_files[i])
#     print('copy file done')
#     data = pd.DataFrame({'target': target_tree_files, 'result': max_nums})
#     data.to_csv('vul-step2-'+str(hold_value)+'.csv')  # 0
#     print('step_1 construct matrix time: ', time_step1)
#     print('step_2 conv and cal time: ', time_step2)
#     return time_step1, time_step2, time_conv


def conv_multi(seed,target):
    file_path = '/root/phase2/target_validate_ast/'
    seed_path = '/root/trees_seed/reentrancy/'
    seed_adj_matrix = get_adj(seed_path+seed)
    target_adj_matrix = get_adj(file_path+target)
    seed_tensor = torch.as_tensor(seed_adj_matrix).double().to(device)
    seed_tensor.unsqueeze_(0).unsqueeze_(0) +9
    with torch.no_grad():
        c = torch.nn.Conv2d(1, 1, kernel_size=len(seed_adj_matrix), stride=1, padding=0, bias=False)
        c.weight.data = seed_tensor
        c = c.to(device)
        # print(len(seed_adj_matrix))
        tmp_max_nums = []
        if len(target_adj_matrix) < len(seed_adj_matrix):
            prob_seed_files.append(target)
            target = None
            max_value = 0
        else:
            try:
                target_tensor = torch.as_tensor(target_adj_matrix).double().to(device)
                target_tensor.unsqueeze_(0).unsqueeze_(0)
                result = c(target_tensor)
                result = result.squeeze()
                # print(torch.sum(seed_tensor == 1).item())
                max_tensor = torch.max(result/(torch.sum(seed_tensor == 1).item()))
                max_value = max_tensor.item()
                # tmp = torch.where(torch.gt(result, 0) & torch.lt(result/(torch.sum(seed_tensor == 1).item()), hold_value),
                #                   torch.zeros_like(result), result)
                # idxs = torch.nonzero(tmp)
                # tmp_max_nums = []
                # if idxs.numel():
                #     for i in range(idxs.shape[0]):
                #         tmp_max_nums.append(tmp[idxs[i][0].item()][idxs[i][1].item()].item() / (torch.sum(seed_tensor == 1).item()))
                # if len(tmp_max_nums) > 0:
                #     pass
                # else:
                #     target = None
            except Exception as e:
                print(f"Error: {e}")
                raise ValueError("An error occurred.") from e
    return target,max_value



def conv_process_multi(seed_adj_matrix,target_adj_matrix,target_ast_path,to_next_ast_path, hold_value):
    seed_adj_np_matrix = np.array([[seed_adj_matrix]])
    seed_tensor = torch.from_numpy(seed_adj_np_matrix)
    seed_tensor = seed_tensor.double().to(device)
    c = torch.nn.Conv2d(1, 1, kernel_size=len(seed_adj_matrix), stride=1, padding=0, bias=False)
    c.weight.data = seed_tensor
    c = c.to(device)
    print(len(seed_adj_matrix))
    to_next_tree_files = []
    time_conv = 0
    max_nums = []
    for k in range(len(target_tree_files)):
        print('start' + target_tree_files[k])
        if len(target_adj_matrix[k]) < len(seed_adj_matrix):
            prob_seed_files.append(target_tree_files[k])
        else:
            try:
                target_adj_np_matrix = np.array([[target_adj_matrix[k]]])
                target_tensor = torch.from_numpy(target_adj_np_matrix)
                target_tensor = target_tensor.double().to(device)
                time_begin = time.perf_counter()
                result = c(target_tensor)

                time_end = time.perf_counter()
                time_conv += (time_end - time_begin)
                result = result.squeeze()
                tmp = torch.where(torch.gt(result, 0) & torch.lt(result, hold_value * (len(seed_adj_matrix) - 1)),
                                  torch.zeros_like(result), result)
                idxs = torch.nonzero(tmp)
                tmp_max_nums = []
                if idxs.numel():
                    for i in range(idxs.shape[0]):
                        tmp_max_nums.append(tmp[idxs[i][0].item()][idxs[i][1].item()].item() / (len(seed_adj_matrix) - 1))
                max_nums.append(tmp_max_nums)
                if len(tmp_max_nums) > 0:
                    to_next_tree_files.append(target_tree_files[k])
                print('--Sim max---')
                print(tmp_max_nums)

            except:
                print(target_tree_files[k] + ' is wrong.')
    if os.path.exists(to_next_ast_path):
        pass
    else:
        os.mkdir(to_next_ast_path)
    for i in range(len(to_next_tree_files)):
        shutil.copyfile(target_ast_path + to_next_tree_files[i], to_next_ast_path + to_next_tree_files[i])

def conv_process(seed_ast_path, target_ast_path, to_next_ast_path, hold_value):
    time1 = time.perf_counter()  #########
    ###
    seed_adj_matrix = get_adj_matrix(seed_ast_path,0)
    target_adj_matrix = get_adj_matrix(target_ast_path, 1)

    for i in range(0,len(seed_adj_matrix)):
        time2 = time.perf_counter()  #########
        # print(target_tree_files)
        # print(prob_seed_files)
        seed_adj_np_matrix = np.array([[seed_adj_matrix[i]]])
        seed_tensor = torch.from_numpy(seed_adj_np_matrix)
        seed_tensor = seed_tensor.double()
        seed_tensor = seed_tensor.to(device)
        # print(seed_tensor)
        ###
        l_seed = len(seed_adj_matrix[i])
        c = torch.nn.Conv2d(1, 1, kernel_size=l_seed, stride=1, padding=0, bias=False)
        c.weight.data = seed_tensor
        c = c.to(device)
        print(len(seed_adj_matrix[0]))
        max_nums = []
        time_step1 = time2 - time1  #########
        time1 = time.perf_counter()
        time_conv = 0
        ###
        for k in range(len(target_tree_files)):
            print('start' + target_tree_files[k])
            if len(target_adj_matrix[k]) < len(seed_adj_matrix[i]):
                prob_seed_files.append(target_tree_files[k])
            else:
                try:
                    target_adj_np_matrix = np.array([[target_adj_matrix[k]]])
                    target_tensor = torch.from_numpy(target_adj_np_matrix)
                    target_tensor = target_tensor.double()
                    target_tensor = target_tensor.to(device)
                    # print(target_tensor)
                    time_begin = time.perf_counter()
                    result = c(target_tensor)
                    time_end = time.perf_counter()
                    time_conv += (time_end - time_begin)
                    result = result.squeeze()
                    # print(result)
                    tmp = torch.where(torch.gt(result, 0) & torch.lt(result, hold_value * (l_seed - 1)),
                                      torch.zeros_like(result), result)
                    # print(torch.nonzero(tmp))
                    # idxs = torch.where(result > hold_value)
                    idxs = torch.nonzero(tmp)
                    # print('--Max sim pos---')
                    # print(idxs)
                    tmp_max_nums = []
                    if idxs.numel():
                        # print(idxs.shape[0], idxs.shape[1])
                        for i in range(idxs.shape[0]):
                            tmp_max_nums.append(tmp[idxs[i][0].item()][idxs[i][1].item()].item() / (l_seed - 1))
                        # print(tmp_max_nums)

                    # maxnum = torch.max(result)
                    # final_maxnum = maxnum.item() / (l_seed - 1)  # 0
                    # max_nums.append(final_maxnum)
                    max_nums.append(tmp_max_nums)
                    if len(tmp_max_nums) > 0:
                        to_next_tree_files.append(target_tree_files[k])
                    print('--Sim max---')
                    print(tmp_max_nums)
                except:
                    print(target_tree_files[k] + ' is wrong.')
        time2 = time.perf_counter()
        time_step2 = time2 - time1
    print('to csv')
    print('num of files :', len(target_tree_files))
    print('to next files :', len(to_next_tree_files))
    if os.path.exists(to_next_ast_path):
        shutil.rmtree(to_next_ast_path)
    os.mkdir(to_next_ast_path)
    for i in range(len(to_next_tree_files)):
        shutil.copyfile(target_ast_path + to_next_tree_files[i], to_next_ast_path + to_next_tree_files[i])
    print('copy file done')
        # data = pd.DataFrame({'target': target_tree_files, 'result': max_nums})
        # data.to_csv('vul-step2-'+str(hold_value)+'.csv')  # 0
    print('step_1 construct matrix time: ', time_step1)
    print('step_2 conv and cal time: ', time_step2)
    return time_step1, time_step2, time_conv