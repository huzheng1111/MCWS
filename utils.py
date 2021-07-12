import torch
import numpy as np


def sequence_padding(inputs, length=None, padding=0):

    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)

def read_file(file_path):
    data = []
    original_data=np.load(file_path, allow_pickle=True).item()
    for index in range(len(original_data["Text"])):
        text = original_data["Text"][index].strip().split()
        l=[]
        for w in text:
            if len(w) == 1 and w in ["，", "：", "。", "！", "；", "、", "（", "）", "《", "》", "-", "‘", ")", "(", "/",
                                     "”", "－", "“", "’", "…", "—"]:
                continue
            else:
                l.append(w)
        data.append((l, original_data["Audio"][index]))
    return data


def eval_sentence(y_pred, y, sentence, tag2index):
    words = "".join(sentence)
    seg_pred = []
    word_pred = ''

    if y is not None:
        word_true = ''
        seg_true = []
        for i in range(len(y)):
            word_true += words[i]
            if y[i] in [tag2index["S"], tag2index["E"]]:
                seg_true.append(word_true)
                word_true = ''
        seg_true_str = ' '.join(seg_true)
    else:
        seg_true_str = None

    for i in range(len(y_pred)):
        word_pred += words[i]
        if y_pred[i] in [tag2index["S"], tag2index["E"]]:
            seg_pred.append(word_pred)
            word_pred = ''
    seg_pred_str = ' '.join(seg_pred)
    return seg_true_str, seg_pred_str


def evaluate(y_pred, y, tag2index):
    cor_num = 0
    p_wordnum = y_pred.count(tag2index["E"]) + y_pred.count(tag2index["S"])
    yt_wordnum = y.count(tag2index["E"]) + y.count(tag2index["S"])
    start = 0
    for i in range(len(y)):
        if y[i] == tag2index["E"] or y[i] == tag2index["S"]:
            flag = True
            for j in range(start, i + 1):
                if y[j] != y_pred[j]:
                    flag = False
            if flag:
                cor_num += 1
            start = i + 1
    return cor_num, p_wordnum, yt_wordnum


def compute_scores(predicted_labels, actual_labels, tag2index):
    correct = 0
    pred_total = 0
    actual_total = 0
    for prediction, tag in zip(predicted_labels, actual_labels):
        cor_num, p_wordnum, yt_wordnum = evaluate(list(prediction), list(tag), tag2index)
        correct += cor_num
        pred_total += p_wordnum
        actual_total += yt_wordnum
    if pred_total == 0:
        precision = 0.0
    else:
        precision = float(correct) / pred_total

    recall = float(correct) / actual_total
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1

def cws_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id, tag2index):
    cor_num = 0
    yt_wordnum = 0
    for y_pred, y, sentence in zip(y_pred_list, y_list, sentence_list):
        start = 0
        for i in range(len(y)):
            if y[i] == tag2index["E"] or y[i] == tag2index["S"]:
                word = ''.join(sentence[0][start:i+1])
                if word in word2id:
                    start = i + 1
                    continue
                flag = True
                yt_wordnum += 1
                for j in range(start, i+1):
                    if y[j] != y_pred[j]:
                        flag = False
                if flag:
                    cor_num += 1
                start = i + 1

    OOV = cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    return OOV

def get_word2id(train_data_path):
    word2id = {}
    index = 0
    texts = np.load(train_data_path, allow_pickle=True).item()["Text"]
    for text in texts:
        text = text.strip().split()
        for word in text:
            if len(word) == 1 and word in ["，", "：", "。", "！", "；", "、", "（", "）", "《", "》", "-", "‘", ")", "(", "/",
                                     "”", "－", "“", "’", "…", "—"]:
                continue
            elif word not in word2id:
                word2id[word] = index
                index += 1
    return word2id


class DataGenerator(object):


    def __init__(self, data, batch_size, tokenizer, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):

        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        while True:
            for d in self.__iter__(False):
                yield d


class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        batch_attention_masks = []
        batch_audio_feature = []
        batch_audio_mask = []
        for is_end, item in self.sample(random):
            token_ids, labels = [], []
            for w in item[0]:
                if len(w) == 1:
                    labels += [0]
                elif len(w)>1:
                    labels += [1] + [2] * (len(w) - 2) + [3]
                for char in w:
                    char_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(char))
                    token_ids += char_token_ids

            assert len(labels) == len(token_ids) == len(item[1])
            assert len(token_ids) <= 510
            batch_audio_feature.append(item[1])
            audio_mask = [1] * (len(token_ids))
            token_ids.insert(0, 101)
            token_ids += [102]

            segment_ids = [0] * len(token_ids)
            attention_mask = [1] * len(token_ids)
            batch_audio_mask.append(audio_mask)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            batch_attention_masks.append(attention_mask)
            if len(batch_token_ids) == self.batch_size or is_end:
                maxlen=0
                for i in range(len(batch_token_ids)):
                    if maxlen<len(batch_token_ids[i]):
                        maxlen=len(batch_token_ids[i])
                batch_token_ids = sequence_padding(batch_token_ids, maxlen)
                batch_segment_ids = sequence_padding(batch_segment_ids, maxlen)
                batch_labels = sequence_padding(batch_labels, maxlen-1)
                batch_attention_masks = sequence_padding(batch_attention_masks, maxlen)
                batch_audio_mask = sequence_padding(batch_audio_mask, maxlen-1)
                batch_audio_feature = sequence_padding(batch_audio_feature, maxlen - 1)
                batch_token_ids = torch.tensor(batch_token_ids)
                batch_segment_ids = torch.tensor(batch_segment_ids)
                batch_labels = torch.tensor(batch_labels)
                batch_attention_masks = torch.tensor(batch_attention_masks)
                batch_audio_mask = torch.tensor(batch_audio_mask)
                batch_audio_feature = torch.tensor(batch_audio_feature, dtype=torch.float32)
                yield [batch_token_ids, batch_attention_masks, batch_segment_ids, batch_audio_feature,
                       batch_audio_mask, batch_labels]
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
                batch_attention_masks = []
                batch_audio_mask = []
                batch_audio_feature = []


