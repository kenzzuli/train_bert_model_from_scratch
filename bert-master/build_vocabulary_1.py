import os
from tokenization import BasicTokenizer
from tqdm import tqdm
from multiprocessing import Pool, Manager
import multiprocessing
import time


class VocabBuilder(object):
    """
    用来构造词表的类
    """

    def __init__(self):
        self.file_full_path_list = self.get_file_path_list()
        self.dict = {"[PAD]": 0,
                     "[UNK]": 1,
                     "[CLS]": 2,
                     "[SEP]": 3,
                     "[MASK]": 4}  # 词表
        self.count_dict = dict()  # 统计每个字符出现次数的词典 {char1:10, char2:12}
        self.sorted_count_list = None  # 按照频数排序后的列表
        self.q_token_list = Manager().Queue()  # 用于存放token_list的序列
        self.q_count_dic = Manager().Queue()  # 用于存放count_dict的序列
        self.q_done = Manager().Queue()  # 用于存放已经完成任务的序列
        self.queue_for_all_dict_only = Manager().Queue()  # 用于存放总的字典的序列，该序列仅存放一个总的字典
        self.queue_for_all_dict_only.put(dict())  # 在序列中放一个空字典

    @staticmethod
    def get_file_path_list():
        """
        获取所有文本的完整路径
        :return:
        """
        data_path = "../corpus_processed"
        file_list = os.listdir(data_path)
        full_file_path_list = [os.path.join(data_path, file) for file in file_list]
        return full_file_path_list

    def tokenize_by_char(self, file):
        """
        按照字符切分句子
        :param file: txt文本
        :return: ['乙', '女', 'は', 'お', '姉', 'さ', 'ま', 'に', '恋', 'し', 'て', 'る', '櫻', 'の', '園', 'の', 'エ', 'ト', 'ワ', 'ー', 'ル']
        """
        basic_tokenizer = BasicTokenizer()
        token_list = []
        with open(file, mode="r", encoding="utf-8") as fin:
            text = fin.read()
            str_list = basic_tokenizer.tokenize(text)
            # print(str_list)
            for str in str_list:
                token_list += list(str)
        # print(token_list)
        self.q_token_list.put(token_list)

    def token_counter(self):
        """
        统计每个字符出现的次数
        :param token_list:
        :return:
        """
        tmp_count_dict = dict()
        token_list = self.q_token_list.get()
        for token in token_list:
            if token not in tmp_count_dict:
                tmp_count_dict[token] = 0
            tmp_count_dict[token] += 1
        self.q_count_dic.put(tmp_count_dict)

    def join_dicts(self):
        """
        将一个进程统计出来的字典与总字典合并
        :return:
        """
        tmp_count_dict = self.q_count_dic.get()
        all_dict = self.queue_for_all_dict_only.get()  # 总字典放在序列中，保证每次只有一个进程可以获取它
        all_keys = tmp_count_dict.keys() | all_dict.keys()
        all_dict = {key: tmp_count_dict.get(key, 0) + all_dict.get(key, 0) for key in all_keys}
        # self.q_done.put(tmp_count_dict)  # 将完成任务的字典放入序列中
        self.q_done.put(1)
        self.queue_for_all_dict_only.put(all_dict)  # 把总字典放回序列中

    def build_vocab(self, min_count=None, max_count=None, vocab_size=None):
        """
        构造词表
        :param min_count: 字符出现的最少次数
        :param max_count: 字符出现的最大次数
        :param vocab_size: 词表中单词的最大数量
        :return:
        """
        self.count_dict = self.queue_for_all_dict_only.get()  # 从序列中获取总字典
        # 对单词频数字典进行筛选，仅保留频数位于[min_count,max_count]的单词
        if isinstance(min_count, int):
            self.count_dict = {key: value for key, value in self.count_dict.items() if min_count <= value}
        if isinstance(max_count, int):
            self.count_dict = {key: value for key, value in self.count_dict.items() if value <= max_count}
            # 对count字典按照频数进行降序排序，再转成列表 [('b', 10), ('a', 9), ('c', 8)]
        self.sorted_count_list = sorted(self.count_dict.items(), key=lambda x: x[1], reverse=True)
        # 限制词表中单词的最大数量
        if isinstance(vocab_size, int):
            # 如果vocab_size<len(sorted_count_list)，则只取前vocab_size个
            if vocab_size < len(self.sorted_count_list):
                self.sorted_count_list = self.sorted_count_list[:vocab_size]
        # 将满足条件的sorted_count_list中的单词，保存到self.dic中
        for token, _ in self.sorted_count_list:
            # if token not in self.dict.keys() and len(self.dict) < vocab_size:  # 加上这一句是为了能够分句传入
            self.dict[token] = len(self.dict)  # dict由{UNK:0,PAD:1} --> {UNK:0,PAD:1,token:2}

    def save_vocab(self):
        dict_path = "../japanese_L-12_H-768_A_12_char/vocab.txt"
        with open(dict_path, mode="w", encoding="utf-8") as fout:
            for token in self.dict.keys():
                fout.write(token + "\n")
        print("Vocabulary Size:%d" % len(self.dict))

    def show_progress_bar(self):
        """做一个进度条"""
        with tqdm(total=len(self.file_full_path_list)) as pbar:
            while True:
                pbar.set_description("Building Vocabulary")
                processed_file_num = self.q_done.qsize()
                pbar.n = processed_file_num
                pbar.refresh()
                if processed_file_num == len(self.file_full_path_list):
                    break

    def multi_process_token_counter(self):
        po = Pool(multiprocessing.cpu_count())
        for file_path in self.file_full_path_list:
            # 往进程池中添加任务
            po.apply_async(self.tokenize_by_char, args=(file_path,))
            po.apply_async(self.token_counter)
            po.apply_async(self.join_dicts)

        self.show_progress_bar()
        po.close()
        po.join()

    def run(self):
        # 多进程词频分词、词频统计、字典合并
        self.multi_process_token_counter()
        # 生成词表
        self.build_vocab()
        # 保存词表
        self.save_vocab()
        print(self.sorted_count_list)


vocab_builder = VocabBuilder()
vocab_builder.run()
