import os
from tqdm import tqdm
work_dir = 'reentrancy/'
node_dir = work_dir + '/node/'
edge_dir = work_dir + '/edge/'

# map user-defined variables to symbolic names(var)
var_list = ['balances[msg.sender]', 'participated[msg.sender]', 'playerPendingWithdrawals[msg.sender]',
            'nonces[msgSender]', 'balances[beneficiary]', 'transactions[transactionId]', 'tokens[token][msg.sender]',
            'totalDeposited[token]', 'tokens[0][msg.sender]', 'accountBalances[msg.sender]', 'accountBalances[_to]',
            'creditedPoints[msg.sender]', 'balances[from]', 'withdrawalCount[from]', 'balances[recipient]',
            'investors[_to]', 'Bal[msg.sender]', 'Accounts[msg.sender]', 'Holders[_addr]', 'balances[_pd]',
            'ExtractDepositTime[msg.sender]', 'Bids[msg.sender]', 'participated[msg.sender]', 'deposited[_participant]',
            'Transactions[TransHash]', 'm_txs[_h]', 'balances[investor]', 'this.balance', 'proposals[_proposalID]',
            'accountBalances[accountAddress]', 'Chargers[id]', 'latestSeriesForUser[msg.sender]',
            'balanceOf[_addressToRefund]', 'tokenManage[token_]', 'milestones[_idMilestone]', 'payments[msg.sender]',
            'rewardsForA[recipient]', 'userBalance[msg.sender]', 'credit[msg.sender]', 'credit[to]', 'round_[_rd]',
            'userPendingWithdrawals[msg.sender]', '[msg.sender]', '[from]', '[to]', '[_to]', "msg.sender", 'abi.decode',
            'abi.encode', 'abi.encodePacked', 'abi.encodeWithSelector', 'abi.encodeCall', 'abi.encodeWithSignature',
            'bytes.concat', 'string.concat', 'block.basefee', 'block.chainid', 'block.coinbase', 'block.difficulty',
            'block.gaslimit', 'block.number', 'block.timestamp','gasleft', 'msg.data', 'msg.sender', 'msg.sig', 'msg.value', 'tx.gasprice', 'tx.origin',
            'assert', 'require', 'revert', 'blockhash', 'keccak256', 'sha256', 'ripemd160',
            'ecrecover', 'addmod', 'mulmod', '(this)', 'super.', 'selfdestruct', 'call.value','now','for','while']

# function limit type
function_limit = ['public', 'private', 'external', 'internal', 'onlyowner', 'onlyOwner']


modifier_list = ['pure', 'view', 'payable', 'constant', 'immutable', 'anonymous', 'indexed', 'virtual',
                 'override']

assignop_list = [' |= ', ' = ', ' ^= ', ' &= ', ' <<= ', ' >>= ', ' += ', ' -= ', ' *= ', ' /= ', ' %= ', ' ++ ',
                 ' -- ', ' + ', ' - ', ' * ', ' / ']

var_op_bool = ['!', '~', '**', '*', '!=', '<', '>', '<=', '>=', '==', '<<', '>>', '||', '&&']

key_list = var_list + function_limit + modifier_list + assignop_list + var_op_bool
final_key_list = []

def  graph_create(target_dir,train_dir) :
    graph_dict = {}
    for file in tqdm(os.listdir(target_dir),desc="graph create"):
        func_word = []
        f = open(target_dir + file,'r',encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            for key_word in key_list:
                if key_word in line and (key_word not in func_word):
                    func_word.append(key_word)
        graph_dict[file.split('.')[0]] = func_word
    for file in os.listdir(train_dir):
        func_word = []
        f = open(train_dir + file,'r',encoding='utf-8')
        lines = f.readlines()
        f.close()
        for line in lines:
            for key_word in key_list:
                if key_word in line and (key_word not in func_word):
                    func_word.append(key_word)
        graph_dict[file.split('.')[0]] = func_word
    return graph_dict


def graph_vec (graph_file,graph_dir):
    f = open(graph_file,'r',encoding='utf-8')
    nodeset = []
    lines = f.readlines()
    f.close()
    f_edge = open(graph_dir + 'edgeset.txt','w',encoding='utf-8')
    for line in tqdm(lines,desc="graph encoding"):
        # print(line)
        start_node = line.split(' ')[0]
        end_node = line.split(' ')[1].strip()
        if start_node not in nodeset:
            nodeset.append(start_node)
        if end_node not in nodeset:
            nodeset.append(end_node)
        start_index = nodeset.index(start_node)
        end_index = nodeset.index(end_node)
        # print(str(start_index) + ' ' + str(end_index))
        f_edge.write(str(start_index) + ' ' + str(end_index) + '\n')
    f_edge.close()
    f_node = open(graph_dir + 'nodeset.txt', 'w', encoding='utf-8')
    for node in nodeset:
        # print(node)
        f_node.write(str(nodeset.index(node)) + ' ' + node +'\n')
    f_node.close()

def graph_vec(graph_file, graph_dir):
    f = open(graph_file, 'r', encoding='utf-8')
    nodeset_dict = {}
    dict_index = 0
    lines = f.readlines()
    f.close()
    f_edge = open(graph_dir + 'edgeset.txt', 'w', encoding='utf-8')
    for line in tqdm(lines, desc="graph encoding"):
        start_node = line.split(' ')[0]
        end_node = line.split(' ')[1].strip()
        if start_node not in nodeset_dict:
            nodeset_dict[start_node] = dict_index
            dict_index += 1
        if end_node not in nodeset_dict:
            nodeset_dict[end_node] = dict_index
            dict_index += 1
        start_index = nodeset_dict[start_node]
        end_index = nodeset_dict[end_node]
        # print(str(start_index) + ' ' + str(end_index))
        f_edge.write(str(start_index) + ' ' + str(end_index) + '\n')
    f_edge.close()
    f_node = open(graph_dir + 'nodeset.txt', 'w', encoding='utf-8')
    # for node in nodeset:
    #     # print(node)
    #     f_node.write(str(nodeset.index(node)) + ' ' + node + '\n')

    for node in nodeset_dict.keys():
        # print(node)
        f_node.write(str(nodeset_dict[node]) + ' ' + node + '\n')
    f_node.close()

# def graph_code (graph_dict , node_index):



def print_graph (graph_dict,graph_dir):
    f = open(graph_dir+'graph.txt','w',encoding='utf-8')
    for file in graph_dict.keys():
        for var in graph_dict[file]:
            var = var.strip()
            f.write(file+ ' ' + var + '\n')
    f.close()


if __name__ == '__main__':
    func_dict = graph_create(key_list, 'reentrancy/func_1025/')
    print_graph(func_dict,'reentrancy/')
    graph_vec('reentrancy/graph.txt', 'reentrancy/node/', 'reentrancy/edge/')

