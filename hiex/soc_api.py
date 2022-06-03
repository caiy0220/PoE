import string

from matplotlib import pyplot as plt
from .soc_algo import _SamplingAndOcclusionAlgo
from .lm import BiGRULanguageModel
from .train_lm import do_train_lm
import os, logging, torch, pickle
from collections import Counter


logger = logging.getLogger(__name__)


def _count2ratio(count_dict, num_word_total):
    dict_ratio = dict()
    for w in count_dict.keys():
        dict_ratio[w] = float(count_dict[w]) / num_word_total
    return dict_ratio

def _fpp(count_dict, total_dict):
    dict_ratio = dict()
    for w in count_dict.keys():
        dict_ratio[w] = float(count_dict[w]) / total_dict[w]
    return dict_ratio

def _get_ratio_diff(ratio0, ratio1, normalized=False):
    base = ratio0 if normalized else 1
    return (ratio0 - ratio1) / base

def plot_top_words(word_value_list, top_k, tokenizer, target_words=[], highlight_words=[], title=''):
    ws = []
    vs = []
    cs = []
    color_list = ['steelblue', 'indianred', 'goldenrod']
    for p in word_value_list[:top_k]:
        word = tokenizer.ids_to_tokens[p[0]]
        ws.append(word)
        vs.append(p[1])
        cs.append(color_picker(word, color_list, [target_words, highlight_words]))
    plt.subplots(figsize=(10, 6))
    plt.xlabel('Frequency')
    plt.ylabel('Term')
    plt.barh(y=ws[::-1], width=vs[::-1], color=cs[::-1])

def color_picker(inp, color_list, rules):
  idx = 0
  for i, r in enumerate(rules):
    idx = i + 1 if inp in r else idx
  return color_list[idx]


class SamplingAndOcclusionExplain:
    def __init__(self, model, configs, tokenizer, output_path, device, lm_dir=None, train_dataloader=None,
                 dev_dataloader=None, vocab=None):
        logger.info('Current version of SOC: 0.000.201')
        self.configs = configs
        self.model = model
        self.lm_dir = lm_dir
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.vocab = vocab
        self.output_path = output_path
        self.device = device
        self.hiex = configs.hiex
        self.tokenizer = tokenizer

        self.lm_model = self.detect_and_load_lm_model()

        self.algo = _SamplingAndOcclusionAlgo(model, tokenizer, self.lm_model, output_path, configs)

        self.use_padding_variant = configs.use_padding_variant
        try:
            self.output_file = open(self.output_path, 'w' if not configs.hiex else 'wb')
        except FileNotFoundError:
            self.output_file = None
        self.output_buffer = []

        # for explanation regularization
        # self.neutral_words_file = configs.neutral_words_file
        # self.neg_suppress_file = configs.neg_suppress_file
        # self.neutral_words_ids = None
        # self.neutral_words = None
        if not self.configs.suppress_weighted:
            self.configs.suppress_fading = 0.    # remove the word from the list immediately after the balance
            self.configs.suppress_increasing = 1.    # do not amplify the weight

        # if configs.neutral_words_file != '':
            # self.mode = 0  # debiasing mode
        self.neutral_words, self.neutral_words_ids = self._loading_words(configs.neutral_words_file)
        #     self.neg_suppress_words = []
        #     self.neg_suppress_words_ids = []
        #     self.pos_suppress_words = []
        #     self.pos_suppress_words_ids = []
        # else:
        #     # self.mode = 1  # debugging mode by suppressing over-sensitive terms in targeted class
        try:
            # self.neg_suppress_words, self.neg_suppress_words_ids = self._loading_words(configs.neg_suppress_file)
            # self.neg_suppress_words, self.neg_suppress_words_ids = self._loading_words(configs.neg_suppress_file)
            self.pos_suppress_words, self.pos_suppress_words_ids = self._loading_words('')
            self.pos_suppress_words, self.pos_suppress_words_ids = self._loading_words('')
            # self.neutral_words = []
            # self.neutral_words_ids = []
        except AttributeError:
            logger.warning('***** Features not exist in given configs, might be using an older version of model *****')
            self.neg_suppress_words, self.neg_suppress_words_ids = dict(), dict()
            self.pos_suppress_words, self.pos_suppress_words_ids = dict(), dict()

        self.word_count_dict = dict()

        self.stop_words, self.stop_words_ids = self._get_stop_words()
        self.count_thresh = 10
        self.filtering_thresh = configs.eta
        self.word_appear_records = dict()
        self.window_size = configs.window_size
        self.window_count = self.window_size * configs.ratio_in_window

    def detect_and_load_lm_model(self):
        if not self.lm_dir:
            self.lm_dir = 'runs/lm/'
        if not os.path.isdir(self.lm_dir):
            os.mkdir(self.lm_dir)

        file_name = None
        for x in os.listdir(self.lm_dir):
            if x.startswith('best'):
                file_name = x
                break
        if not file_name:
            self.train_lm()
            for x in os.listdir(self.lm_dir):
                if x.startswith('best'):
                    file_name = x
                    break
        lm_model = torch.load(open(os.path.join(self.lm_dir,file_name), 'rb'))
        return lm_model

    def train_lm(self):
        logger.info('Missing pretrained LM. Now training')
        model = BiGRULanguageModel(self.configs, vocab=self.vocab, device=self.device).to(self.device)
        do_train_lm(model, lm_dir=self.lm_dir, lm_epochs=20,
                    train_iter=self.train_dataloader, dev_iter=self.dev_dataloader)

    def word_level_explanation_bert(self, input_ids, input_mask, segment_ids, label=None):
        # requires batch size is 1
        # get sequence length
        i = 0
        while i < input_ids.size(1) and input_ids[0,i] != 0: # pad
            i += 1
        inp_length = i
        # do not explain [CLS] and [SEP]
        spans, scores = [], []
        for i in range(1, inp_length-1, 1):
            span = (i, i)
            spans.append(span)
            if not self.use_padding_variant:
                score = self.algo.do_attribution(input_ids, input_mask, segment_ids, span, label)
            else:
                score = self.algo.do_attribution_pad_variant(input_ids, input_mask, segment_ids, span, label)
            scores.append(score)
        inp = input_ids.view(-1).cpu().numpy()
        s = self.algo.repr_result_region(inp, spans, scores)
        self.output_file.write(s + '\n')

    def hierarchical_explanation_bert(self, input_ids, input_mask, segment_ids, label=None):
        tab_info = self.algo.do_hierarchical_explanation(input_ids, input_mask, segment_ids, label)
        self.output_buffer.append(tab_info)
        # currently store a pkl after explaining each instance
        self.output_file = open(self.output_path, 'w' if not self.hiex else 'wb')
        pickle.dump(self.output_buffer, self.output_file)
        self.output_file.close()

    def _loading_words(self, pth):
        if pth == '':
            return dict(), dict()
        with open(pth) as f:
            words = dict()
            words_ids = dict()
            for line in f.readlines():
                segs = line.strip().split('\t')
                word = segs[0]
                val = float(segs[1]) if self.configs.suppress_weighted else 1.
                canonical = self.tokenizer.tokenize(word)
                if len(canonical) > 1:
                    canonical.sort(key=lambda x: -len(x))
                    print(canonical)
                words[word] = val
                words_ids[self.tokenizer.vocab[word]] = val
            assert words
        return words, words_ids

    def _get_stop_words(self):
        stop_words = {'[CLS]', '[PAD]', '[SEP]'}
        for pc in string.punctuation:
            stop_words.add(pc)

        stop_words_ids = set()
        for s in stop_words:
            try:
                stop_words_ids.add(self.vocab[s])
            except KeyError:
                logger.warning('Cannot find target word in vocab:\t{}'.format(s))
        return stop_words, stop_words_ids

    def _words_count(self, corpus):
        words_ids = []
        for t in corpus:
            words_ids += [w for w in t if w not in self.stop_words_ids]
        words_ids_count = Counter(words_ids).most_common()
        return words_ids_count, len(words_ids)

    def _filter_minimal_count(self, count_dict):
        total_count = 0
        cp = count_dict.copy()
        for w in count_dict.keys():
            if count_dict[w] <= self.count_thresh:
                del cp[w]
            else:
                total_count += count_dict[w]
        return cp, total_count

    def get_global_words_count(self, corpus):
        counter, _ = self._words_count(corpus)
        for wid, cnt in counter:
            self.word_count_dict[wid] = cnt

    def get_suppress_words(self):
        return self.neg_suppress_words.copy()

    def get_neutral_words(self):
        return self.neutral_words.copy()

    def update_suppress_words_lazy(self, wrong_li, right_li, verbose=0, allow_change=False):
        fns, fps, tns, tps = [], [], [], []
        for idx in range(len(wrong_li[0])):
            if wrong_li[1][idx] == 1:
                # the ground truth is positive, indicating the instance is false negative instance
                fns.append(wrong_li[0][idx].numpy().tolist())
            else:
                fps.append(wrong_li[0][idx].numpy().tolist())

        for idx in range(len(right_li[0])):
            if right_li[1][idx] == 1:
                # the ground truth is positive, indicating the instance is false negative instance
                tps.append(right_li[0][idx].numpy().tolist())
            else:
                tns.append(right_li[0][idx].numpy().tolist())

        tnps = tns + tps

        normalized = True

        fns_word_count_li, fns_word_num_total = self._words_count(fns)
        fps_word_count_li, fps_word_num_total = self._words_count(fps)
        tnps_word_count_li, tnps_word_num_total = self._words_count(tnps)

        # TODO: filter out rear terms directly while computing FPP
        fns_word_count, fns_word_num_total = self._filter_minimal_count(dict(fns_word_count_li))
        fps_word_count, fps_word_num_total = self._filter_minimal_count(dict(fps_word_count_li))
        tnps_word_count, tnps_word_num_total = self._filter_minimal_count(dict(tnps_word_count_li))

        # fns_word_ratio = _count2ratio(fns_word_count, fns_word_num_total)
        # fps_word_ratio = _count2ratio(fps_word_count, fps_word_num_total)
        # tnps_word_ratio = _count2ratio(tnps_word_count, tnps_word_num_total)
        fns_word_ratio = _fpp(fns_word_count, self.word_count_dict)
        fps_word_ratio = _fpp(fps_word_count, self.word_count_dict)
        tnps_word_ratio = _fpp(tnps_word_count, self.word_count_dict)
        sorted_diff_fns_tnps = sorted(fns_word_ratio.items(), key=lambda item: item[1])[::-1]
        sorted_diff_fps_tnps = sorted(fps_word_ratio.items(), key=lambda item: item[1])[::-1]

        if allow_change:
            new_suppress_words_ids = []
            cnt = 0
            for p in sorted_diff_fps_tnps:
                if p[1] <= self.filtering_thresh:
                    break
                cnt += 1

                if verbose:
                    logger.info('{:<12}: {:.3f}, [{:<3}, {:<5}]'.format(self.tokenizer.ids_to_tokens[p[0]], p[1], fps_word_count[p[0]], self.word_count_dict[p[0]]))
                target = p[0]
                new_suppress_words_ids.append(target)
                if target not in self.word_appear_records:
                    self.word_appear_records[target] = [1]
                else:
                    self.update_word_appear_records(target, 1)

            if verbose:
                logger.info('#{:<4} words added this round'.format(cnt))

            for w_ids in self.word_appear_records.keys():
                if w_ids not in new_suppress_words_ids:
                    self.update_word_appear_records(w_ids, 0)
            self._update_suppress_words()
            # logger.info('------- Current Suppressing List --------')
            # logger.info(self.neg_suppress_words)
        return fps_word_ratio, tnps_word_ratio, fns_word_ratio

    def _update_suppress_words(self):
        word_counts_dict = self._get_word_counts()
        for w_ids in word_counts_dict.keys():
            if word_counts_dict[w_ids] == 0:
                self.word_appear_records.pop(w_ids)

            if word_counts_dict[w_ids] >= self.window_count:
                w = self.tokenizer.ids_to_tokens[w_ids]
                self.neg_suppress_words_ids[w_ids] = 1.
                self.neg_suppress_words[w] = 1.

    def filter_suppress_words(self, ws):
        for w in ws:
            w_id = self.tokenizer.vocab[w]
            del self.neg_suppress_words_ids[w_id]
            del self.neg_suppress_words[w]

    def _get_word_counts(self):
        word_count_dict = dict()
        to_be_removed = []
        for w_ids in self.word_appear_records.keys():
            word_count_dict[w_ids] = sum(self.word_appear_records[w_ids])
        return word_count_dict

    def update_word_appear_records(self, w_ids, v):
        self.word_appear_records[w_ids].append(v)
        if len(self.word_appear_records[w_ids]) > self.window_size:
            self.word_appear_records[w_ids].pop(0)

    def update_suppress_words(self, wrong_li, right_li, verbose=0):
        fns, fps, tns, tps = [], [], [], []
        for idx in range(len(wrong_li[0])):
            if wrong_li[1][idx] == 1:
                # the ground truth is positive, indicating the instance is false negative instance
                fns.append(wrong_li[0][idx].numpy().tolist())
            else:
                fps.append(wrong_li[0][idx].numpy().tolist())

        for idx in range(len(right_li[0])):
            if right_li[1][idx] == 1:
                # the ground truth is positive, indicating the instance is false negative instance
                tps.append(right_li[0][idx].numpy().tolist())
            else:
                tns.append(right_li[0][idx].numpy().tolist())

        tnps = tns + tps

        normalized = True

        fns_word_count_li, fns_word_num_total = self._words_count(fns)
        fps_word_count_li, fps_word_num_total = self._words_count(fps)
        tnps_word_count_li, tnps_word_num_total = self._words_count(tnps)

        fns_word_count, fns_word_num_total = self._filter_minimal_count(dict(fns_word_count_li))
        fps_word_count, fps_word_num_total = self._filter_minimal_count(dict(fps_word_count_li))
        tnps_word_count, tnps_word_num_total = self._filter_minimal_count(dict(tnps_word_count_li))

        fns_word_ratio = _count2ratio(fns_word_count, fns_word_num_total)
        fps_word_ratio = _count2ratio(fps_word_count, fps_word_num_total)
        tnps_word_ratio = _count2ratio(tnps_word_count, tnps_word_num_total)

        word_ratio_diff_fns_tnps = dict()
        for w in fns_word_ratio.keys():
            if w not in tnps_word_ratio:
                word_ratio_diff_fns_tnps[w] = 1.
                # continue
            else:
                word_ratio_diff_fns_tnps[w] = _get_ratio_diff(fns_word_ratio[w], tnps_word_ratio[w], normalized)
        sorted_diff_fns_tnps = sorted(word_ratio_diff_fns_tnps.items(), key=lambda item: item[1])[::-1]

        word_ratio_diff_fps_tnps = dict()
        for w in fps_word_ratio.keys():
            if w not in tnps_word_ratio:
                word_ratio_diff_fps_tnps[w] = 1.
                # continue
            else:
                word_ratio_diff_fps_tnps[w] = _get_ratio_diff(fps_word_ratio[w], tnps_word_ratio[w], normalized)
        sorted_diff_fps_tnps = sorted(word_ratio_diff_fps_tnps.items(), key=lambda item: item[1])[::-1]

        if verbose:
            target_words = ['white', 'black', 'jew', 'muslims', 'jews', 'islam']
            new_words = ['blacks', 'whites', 'muslim', 'women', 'obama']
            plot_top_words(sorted_diff_fns_tnps, 30, self.tokenizer, target_words + new_words, [], title='False negative')
            plot_top_words(sorted_diff_fps_tnps, 30, self.tokenizer, target_words + new_words, [], title='False positive')
            plt.show()

        new_suppress_words_ids = []
        for p in sorted_diff_fps_tnps:
            if p[1] <= self.filtering_thresh:
                break
            new_suppress_words_ids.append(p[0])

            w = self.tokenizer.ids_to_tokens[p[0]]

            if p[0] in self.neg_suppress_words_ids:
                self.neg_suppress_words_ids[p[0]] *= self.configs.suppress_increasing
                update_val = min(self.neg_suppress_words_ids[p[0]], self.configs.suppress_higher_thresh)
                self.neg_suppress_words_ids[p[0]] = update_val
                self.neg_suppress_words[w] = update_val
            else:
                self.neg_suppress_words_ids[p[0]] = p[1] if self.configs.suppress_weighted else 1.
                self.neg_suppress_words[w] = p[1] if self.configs.suppress_weighted else 1.

        for w_ids in list(self.neg_suppress_words_ids.keys()):
            if w_ids not in new_suppress_words_ids:
                w = self.tokenizer.ids_to_tokens[w_ids]
                self.neg_suppress_words_ids[w_ids] *= self.configs.suppress_fading
                self.neg_suppress_words[w] *= self.configs.suppress_fading
                if self.neg_suppress_words_ids[w_ids] <= self.configs.suppress_lower_thresh:
                    self.neg_suppress_words_ids.pop(w_ids)
                    self.neg_suppress_words.pop(w)

    '''
    def _initialize_neutral_words(self):
        f = open(self.neutral_words_file)
        neutral_words = []
        neutral_words_ids = set()
        for line in f.readlines():
            word = line.strip().split('\t')[0]
            canonical = self.tokenizer.tokenize(word)
            if len(canonical) > 1:
                canonical.sort(key=lambda x: -len(x))
                print(canonical)
            word = canonical[0]
            neutral_words.append(word)
            neutral_words_ids.add(self.tokenizer.vocab[word])
        self.neutral_words = neutral_words
        self.neutral_words_ids = neutral_words_ids
        assert neutral_words
    '''

    def suppress_explanation_loss(self, input_ids_batch, input_mask_batch, segment_ids_batch, label_ids_batch,
                                 do_backprop=False):
        # TODO: only for negative words by now
        if len(self.neg_suppress_words_ids) == 0:
            return 0., 0.

        batch_size = input_ids_batch.size(0)
        word_scores = []
        for b in range(batch_size):
            input_ids, input_mask, segment_ids, label_ids = input_ids_batch[b], \
                                                            input_mask_batch[b], \
                                                            segment_ids_batch[b], \
                                                            label_ids_batch[b]
            nw_positions = []
            if label_ids == 1:
                continue
            for i in range(len(input_ids)):
                word_id = input_ids[i].item()
                if word_id in self.neg_suppress_words_ids:
                    nw_positions.append(i)

            # only generate explanations for targeted words
            for i in range(len(input_ids)):
                word_id = input_ids[i].item()
                if word_id in self.neg_suppress_words_ids:
                    x_region = (i, i)
                    if not self.configs.use_padding_variant:
                        score = self.algo.do_attribution(input_ids, input_mask, segment_ids, x_region, label_ids,
                                                         return_variable=True, additional_mask=nw_positions)
                    else:
                        score = self.algo.do_attribution_pad_variant(input_ids, input_mask, segment_ids,
                                                                     x_region, label_ids, return_variable=True,
                                                                     additional_mask=nw_positions)
                    if label_ids == 0 and self.configs.reg_balanced:
                        weight = self.configs.negative_weight
                    else:
                        weight = 1.
                    # score = self.configs.reg_strength * (score ** 2) * weight
                    weight *= self.configs.reg_strength
                    score = self.neg_suppress_words_ids[word_id] * (score ** 2) * weight

                    if do_backprop:
                        score.backward()

                    word_scores.append(score.item())

        if word_scores:
            return sum(word_scores), len(word_scores)
        else:
            return 0., 0

    def compute_explanation_loss(self, input_ids_batch, input_mask_batch, segment_ids_batch, label_ids_batch,
                                 do_backprop=False):
        # if self.neutral_words is None:
        #     self._initialize_neutral_words()
        batch_size = input_ids_batch.size(0)
        neutral_word_scores, cnt = [], 0
        for b in range(batch_size):
            input_ids, input_mask, segment_ids, label_ids = input_ids_batch[b], \
                                                            input_mask_batch[b], \
                                                            segment_ids_batch[b], \
                                                            label_ids_batch[b]
            nw_positions = []
            for i in range(len(input_ids)):
                word_id = input_ids[i].item()
                if word_id in self.neutral_words_ids:
                    nw_positions.append(i)
            # only generate explanations for neutral words
            for i in range(len(input_ids)):
                word_id = input_ids[i].item()
                if word_id in self.neutral_words_ids:
                    x_region = (i, i)
                    # score = self.algo.occlude_input_with_masks_and_run(input_ids, input_mask, segment_ids,
                    #                                                    [x_region], nb_region, label_ids,
                    #                                                    return_variable=True)
                    if not self.configs.use_padding_variant:
                        score = self.algo.do_attribution(input_ids, input_mask, segment_ids, x_region, label_ids,
                                                         return_variable=True, additional_mask=nw_positions)
                    else:
                        score = self.algo.do_attribution_pad_variant(input_ids, input_mask, segment_ids,
                                                                     x_region, label_ids, return_variable=True,
                                                                     additional_mask=nw_positions)
                    if label_ids == 0 and self.configs.reg_balanced:
                        weight = self.configs.negative_weight
                    else:
                        weight = 1.
                    score = self.configs.reg_strength * (score ** 2) * weight
                    # score = min(1., score)

                    if do_backprop:
                        score.backward()

                    neutral_word_scores.append(score.item())

        if neutral_word_scores:
            return sum(neutral_word_scores), len(neutral_word_scores)
        else:
            return 0., 0
