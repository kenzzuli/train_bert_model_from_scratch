import os
from tokenization import BasicTokenizer
from tqdm import tqdm
from multiprocessing import Pool, Manager


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
        self.q_count_dic = Manager().Queue()  # 用于存放所有线程的count_dic
        self.q_count_dic_done = Manager().Queue()  # 用于存放已经完成字典合并的count_dic序列

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

    def tokenize_and_count(self, file):
        """
        按照字符切分句子，并统计词频
        :param file: txt文本
        :return: ['乙', '女', 'は', 'お', '姉', 'さ', 'ま', 'に', '恋', 'し', 'て', 'る', '櫻', 'の', '園', 'の', 'エ', 'ト', 'ワ', 'ー', 'ル']
        """
        basic_tokenizer = BasicTokenizer(do_lower_case=False)  # 必须加上do_lower_case=False,这样じ才不会变成し
        token_list = []
        with open(file, mode="r", encoding="utf-8") as fin:
            text = fin.read()
            str_list = basic_tokenizer.tokenize(text)
            for str in str_list:
                token_list += list(str)
        # 统计
        tmp_count_dict = dict()
        for token in token_list:
            if token not in tmp_count_dict:
                tmp_count_dict[token] = 0
            tmp_count_dict[token] += 1
        self.q_count_dic.put(tmp_count_dict)  # 将统计结果放入序列

    def join_dicts(self):
        """
        使用一个单独的进程，将所有进程统计出来的字典与总字典合并
        :return:
        """
        while True:
            tmp_count_dict = self.q_count_dic.get()  # 从队列获取统计出来的字典
            all_keys = tmp_count_dict.keys() | self.count_dict.keys()
            self.count_dict = {key: tmp_count_dict.get(key, 0) + self.count_dict.get(key, 0) for key in all_keys}
            self.q_count_dic_done.put(1)  # 将完成任务的字典放入序列中，只放字典的长度即可，不必放整个字典
            if self.q_count_dic_done.qsize() == len(self.file_full_path_list):
                self.q_count_dic.put(self.count_dict)  # 通过序列传递最终的结果
                break

    def build_vocab(self, min_count=None, max_count=None, vocab_size=None):
        """
        构造词表
        :param min_count: 字符出现的最少次数
        :param max_count: 字符出现的最大次数
        :param vocab_size: 词表中单词的最大数量
        :return:
        """
        self.count_dict = self.q_count_dic.get_nowait()  # 从序列中获取最终的字典
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
        dict_path = "./vocab.txt"
        with open(dict_path, mode="w", encoding="utf-8") as fout:
            for token in self.dict.keys():
                fout.write(token + "\n")
        print("Vocabulary Size:%d" % len(self.dict))

    def show_progress_bar(self):
        """做一个进度条"""
        with tqdm(total=len(self.file_full_path_list)) as pbar:
            while True:
                pbar.set_description("Building Vocabulary")
                processed_file_num = self.q_count_dic_done.qsize()
                pbar.n = processed_file_num
                pbar.refresh()
                if processed_file_num == len(self.file_full_path_list):
                    break

    def multi_process_token_counter(self):
        po = Pool()
        po.apply_async(self.join_dicts)  # 添加合并字典任务
        for file_path in self.file_full_path_list:
            # 添加分词并统计任务
            po.apply_async(self.tokenize_and_count, args=(file_path,))
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


if __name__ == '__main__':
    vocab_builder = VocabBuilder()
    vocab_builder.run()
