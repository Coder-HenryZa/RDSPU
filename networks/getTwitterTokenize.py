from networks.word2sequence import Word2Sequence
import pickle
import re
import emoji
import pandas
stopwords = set([i.strip() for i in open(r'../datasets/stopwords.txt', 'r', encoding="utf-8").readlines()])

def tokenize(sentence):
    sentence = emoji.demojize(sentence)
    fileters = ['!', '"', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '\[', '\\', '\]', '^', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“']
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<.*?>", " ", sentence)
    sentence = re.sub("<br />", " ", sentence)
    # 把表情去掉
    # sentence = re.sub("[^\\u0000-\\uFFFF]", ' ', sentence)
    # 把http替换为 url
    sentence = re.sub("(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", 'url', sentence)
    # 把@PramilaJayapal 替换为@xx
    sentence = re.sub("@[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", '@xxx', sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    sentence = sentence.replace("’s", " is")
    sentence = sentence.replace("’m", " am")
    sentence = sentence.replace("n’t", " not")
    sentence = sentence.replace("n't", " not")
    sentence = sentence.replace("…", "")
    result = [i for i in sentence.split(" ") if len(i) > 0 and i not in stopwords]

    return result

def loadTree(dataname):
    treeDic = {}
    for line in open('../datasets/' + dataname + '/data.TD_RvNN.vol_5000.txt'):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}

    return treeDic

if __name__ == '__main__':


    dataname = 'Twitter15' # Twitter15, Twitter16

    ws = Word2Sequence()

    statuses = pandas.read_csv('../datasets/' + dataname + '/' + dataname + '_Statuses.csv', sep='\t', encoding='utf-8')
    statuses['twitter_id'] = statuses['twitter_id'].astype(str)
    statuses_dict = statuses.set_index('twitter_id')['20_status'].to_dict()


    treeDic = loadTree(dataname)
    id_list = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= 2 and len(treeDic[id]) <= 100000, list(statuses_dict.keys())))
    for key in id_list:
        value = eval(statuses_dict[key])
        if type(value[0]) != list:
            for i in value:
                if type(i) != list:
                    sentence = tokenize(i)
                    ws.fit(sentence[:35])

    ws.build_vocab(min_count=10)
    pickle.dump(ws, open('../datasets/' + dataname + '_ws.pkl', 'wb'))
    print(len(ws))