# coding: UTF-8  #设置编码
import os
import shutil
import tqdm


final_file_path = 'new_train1.txt'  # 新的txt
reference_file_path = 'new_train.txt'  # txt文件

final = open(final_file_path, mode='w')

file = open(reference_file_path, 'r', encoding='UTF-8')
for line in tqdm.tqdm(file.readlines()):
    line = line[:-2]
    line = line.rstrip()
    final.write(line + '\n')

final.close()






