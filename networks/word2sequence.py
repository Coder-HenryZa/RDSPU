class Word2Sequence():
    #标记特殊字符
    UNK_TAG = "UNK"
    #填充字符
    PAD_TAG = "PAD"
    #异常字符
    ABN_TAG = "ABN"

    UNK = 2
    PAD = 1
    ABN = 0
    def __init__(self):
        self.dict = {self.UNK_TAG:self.UNK,self.PAD_TAG:self.PAD, self.ABN_TAG:self.ABN}
        self.count = {}

    def fit(self,sentence):
        """
        :param sentences:[[word1,word2,word3],[word1,word3,wordn..],...]
        :param min_count: 最小出现的次数
        :param max_count: 最大出现的次数
        :param max_feature: 总词语的最大数量
        :return:
        """

        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1




    def build_vocab(self, min_count=None, max_count=None, max_feature=None):
        # 比最小的数大，和比最大的数小需要执行
        if min_count is not None:
            self.count = {key:value for key,value in self.count.items() if value >= min_count}
        if max_count is not None:
            self.count = {key:value for key,value in self.count.items() if value <= max_count}

        # 限制词语最大数量
        if max_feature is not None:
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_feature])

        # 构建self.dict()  {词：编号，词：编号....}
        for word in self.count.keys():
            self.dict[word] = len(self.dict)  # 获取每个词及生成每个词对应的编号

        # 构建反转字典，self.inverse_dict {编号：词，编号：词}
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self,sentence,max_len = None):
        '''
        把句子转化为数字序列,并且使得序列长度一致
        :param sentense: [str,str,,,,,,,,,,]
        :return: [num,num,num,,,,,,,]
        '''
        if max_len is not None and len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))
        return [self.dict.get(i,0) for i in sentence]


    def inverse_transform(self,incides):
        '''
        把数字序列转化为字符
        :param incides: [num,num,num,,,,,,,,]
        :return: [str,str,str,,,,,,,]
        '''
        return [self.inverse_dict.get(i,"UNK") for i in incides]


    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    w2s = Word2Sequence()
    w2s.fit(["你", "好", "么"])
    w2s.build_vocab()

    print('123456')
    print(w2s.dict)
    print(w2s.transform(["你","好","嘛"],max_len=5))
    print(w2s.inverse_transform([2, 3, 0, 1, 1]))