import os


def read_file(file_path, encoding):
    with open(file_path, mode="r", encoding=encoding) as fin:
        text = fin.read()
        return text


def write_file(file_path, encoding, text):
    with open(file_path, mode="a", encoding=encoding) as fout:
        fout.write(text)


def split_by_batch(file_list, batch_size):
    result = []
    while len(file_list) > batch_size:
        tmp_list = file_list[:batch_size]
        result.append(tmp_list)
        file_list = file_list[batch_size:]
    result.append(file_list)
    return result


if not os.path.exists("../corpus_concatenated"):  # 创建文件夹，存储处理好的文件
    os.mkdir("../corpus_concatenated")
os.chdir("../corpus_processed")  # 更改目录到原始语料文件夹
file_list = os.listdir("./")
file_list.sort()
batch_size = 2000
result_list = split_by_batch(file_list, batch_size)
for batch_num, batch_list in enumerate(result_list):
    file_out = "../corpus_concatenated/pretrain_data_batch-{}.txt".format(batch_num)
    for file in batch_list:
        text = read_file(file, "utf8")
        write_file(file_out, "utf8", text)
        write_file(file_out, "utf8", "\n\n")
