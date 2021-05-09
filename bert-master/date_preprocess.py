# 对原文本进行批量处理，确保每行仅有一句话，去除多余的换行符
import os
import re

if not os.path.exists("../corpus_processed"):  # 创建文件夹，存储处理好的文件
    os.mkdir("../corpus_processed")
os.chdir("../corpus_raw")  # 更改目录到原始语料文件夹

file_list = os.listdir()  # 获取文件列表
i = 1
text = ""
for file in file_list:
    with open(file, "r", encoding="utf8") as fin:
        text = fin.read()
        while re.search("\n\s", text, re.M):  # 匹配换行符 \s是匹配空白字符，包括换行制表空格等字符
            text = re.sub("\n\s", "\n", text)  # 替换
    with open("../corpus_processed/pretrain_data_{}.txt".format(i), "w", encoding="utf8") as fout: # 写入文件
        fout.write(text)
        i += 1







