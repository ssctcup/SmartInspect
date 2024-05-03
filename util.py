import re
import os
import uuid
import shutil


def remove_comment(inputFile, outputFile):
    fdr = open(inputFile, 'r', encoding="utf-8")
    fdw = open(outputFile, 'w', encoding="utf-8")
    _map = {}
    outstring = ''
    line = fdr.readline()
    while line:
        while True:
            m = re.compile('\".*\"', re.S)
            _str = m.search(line)
            if None == _str:
                outstring += line
                break
            key = str(uuid.uuid1())
            m = re.compile('\".*\"', re.S)
            outtmp = re.sub(m, key, line, 1)
            line = outtmp
            _map[key] = _str.group(0)
        line = fdr.readline()
    m = re.compile(r'//.*')
    outtmp = re.sub(m, ' ', outstring)
    outstring = outtmp

    m = re.compile(r'/\*.*?\*/', re.S)
    outtmp = re.sub(m, ' ', outstring)
    outstring = outtmp

    for key in _map.keys():
        outstring = outstring.replace(key, _map[key])

    fdw.write(outstring)
    fdw.close()


def split_function(filepath):
    function_list = []
    stack_list = [] #store { in text
    f = open(filepath, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    tmp = []
    for line in lines:
        text = line.strip()
        if len(text) > 0 and text != "\n":
            if (text.split()[0] == "function" or text.split()[0] == "function()") and ";" not in text:
                tmp = [text]
                if "{" in text:
                    stack_list.append("{")
                if "}" in text:
                    stack_list.pop()
                    tmp = []
            elif len(tmp) > 0:
                tmp.append(text)
                if "{" in text:
                    stack_list.append("{")
                if "}" in text:
                    if len(stack_list) == 1:
                        function_list.append(tmp)
                        tmp = []
                    if len(stack_list) > 0:
                        stack_list.pop()
                    else:
                        tmp = []
    return function_list


def get_tree(func_dir,AST_dir,flag):
    if os.path.exists(AST_dir):
        shutil.rmtree(AST_dir)
    os.mkdir(AST_dir)
    cmd_str = "sh sparse_tree.sh "+ func_dir  +" "+ AST_dir + " "+ flag
    cmd_sys = os.system(cmd_str)





