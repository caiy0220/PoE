from .soc_algo import _SamplingAndOcclusionAlgo
from .lm import BiGRULanguageModel
from .train_lm import do_train_lm
import os
import logging
import torch
import pickle


logger = logging.getLogger(__name__)
VERSION = 'Current version of SOC: 0.001.001'


def _count2ratio(count_dict, num_word_total):
    dict_ratio = dict()
    for w in count_dict.keys():
        dict_ratio[w] = float(count_dict[w]) / num_word_total
    return dict_ratio

def _get_ratio_diff(ratio0, ratio1, normalized=False):
    base = ratio0 if normalized else 1
    return (ratio0 - ratio1) / base

def color_picker(inp, color_list, rules):
  idx = 0
  for i, r in enumerate(rules):
    idx = i + 1 if inp in r else idx
  return color_list[idx]


class SamplingAndOcclusionExplain:
    def __init__(self, model, configs, tokenizer, processor, output_path, device, lm_dir=None):
        logger.info(VERSION)
        self.configs = configs
        self.model = model
        self.lm_dir = lm_dir
        self.output_path = output_path
        self.device = device
        self.hiex = configs.hiex
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab

        self.lm_model = self.detect_and_load_lm_model(processor)

        self.algo = _SamplingAndOcclusionAlgo(model, tokenizer, self.lm_model, output_path, configs)

        self.use_padding_variant = configs.use_padding_variant
        try:
            self.output_file = open(self.output_path, 'w' if not configs.hiex else 'wb')
        except FileNotFoundError:
            self.output_file = None
        self.output_buffer = []

        if not self.configs.suppress_weighted:
            self.configs.suppress_fading = 0.    # remove the word from the list immediately after the balance
            self.configs.suppress_increasing = 1.    # do not amplify the weight

        # self.neutral_words, self.neutral_words_ids = self._loading_words(configs.neutral_words_file)
        # try:
        #     self.neg_suppress_words, self.neg_suppress_words_ids = self._loading_words('')
        #     self.pos_suppress_words, self.pos_suppress_words_ids = self._loading_words('')
        # except AttributeError:
        #     logger.warning('***** Features not exist in given configs, might be using an older version of model *****')
        #     self.neg_suppress_words, self.neg_suppress_words_ids = dict(), dict()
        #     self.pos_suppress_words, self.pos_suppress_words_ids = dict(), dict()

        # self.word_count_dict = dict()

        # self.stop_words, self.stop_words_ids = self._get_stop_words()
        # self.count_thresh = configs.count_thresh
        # self.filtering_thresh = configs.eta
        # self.word_appear_records = dict()
        # self.window_size = configs.window_size
        # self.window_count = self.window_size * configs.ratio_in_window

    def detect_and_load_lm_model(self, processor):
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
            self.train_lm(processor)
            for x in os.listdir(self.lm_dir):
                if x.startswith('best'):
                    file_name = x
                    break
        lm_model = torch.load(open(os.path.join(self.lm_dir, file_name), 'rb'), weights_only=False)
        return lm_model

    def train_lm(self, processor):
        logger.info('Missing pretrained LM. Now training')
        model = BiGRULanguageModel(self.configs, vocab=self.vocab, device=self.device).to(self.device)
        train_dataloader = processor.get_dataloader('train', self.configs.train_batch_size)
        dev_dataloader = processor.get_dataloader('dev', self.configs.train_batch_size)
        do_train_lm(model, lm_dir=self.lm_dir, lm_epochs=20, train_iter=train_dataloader, dev_iter=dev_dataloader)

    def word_level_explanation_bert(self, input_ids, input_mask, segment_ids, label=None):
        # requires batch size is 1
        # get sequence length
        i = 0
        while i < input_ids.size(1) and input_ids[0, i] != 0:    # pad
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

    def suppress_explanation_loss(self, batch, targets, ignore_label=-1, do_backprop=False):
        if len(targets) == 0:
            return 0., 0.

        word_scores = []
        for b in range(batch[0].size(0)):
            input_ids, input_mask, segment_ids, label_ids = [e[b] for e in batch]
            if label_ids == ignore_label:
                continue

            nw_positions = []
            for i in range(len(input_ids)):
                word_id = input_ids[i].item()
                if word_id in targets:
                    nw_positions.append(i)

            # only generate explanations for targeted words
            for i in range(len(input_ids)):
                word_id = input_ids[i].item()
                if word_id in targets:
                    x_region = (i, i)
                    if not self.configs.use_padding_variant:
                        score = self.algo.do_attribution(input_ids, input_mask, segment_ids, x_region,
                                                         label_ids, return_variable=True, additional_mask=nw_positions)
                    else:
                        score = self.algo.do_attribution_pad_variant(input_ids, input_mask, segment_ids,
                                                                     x_region, label_ids, return_variable=True, additional_mask=nw_positions)
                    score = targets[word_id] * (score ** 2) * self.configs.reg_strength

                    if do_backprop:
                        score.backward()

                    word_scores.append(score.item())

        if word_scores:
            return sum(word_scores), len(word_scores)
        else:
            return 0., 0

    """
    ================================================================
    |                       Deprecated                             |
    ================================================================
    """
    """
    def get_suppress_words(self):
        return self.neg_suppress_words.copy()

    def get_neutral_words(self):
        return self.neutral_words.copy()

    def _words_count(self, corpus):
        words_ids = []
        for t in corpus:
            words_ids += [w for w in t if w not in self.stop_words_ids]
        words_ids_count = Counter(words_ids).most_common()
        return words_ids_count, len(words_ids)

    def get_global_words_count(self, corpus):
        counter, _ = self._words_count(corpus)
        for wid, cnt in counter:
            self.word_count_dict[wid] = cnt

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

        fns_count, _ = self._words_count(fns)
        fps_count, _ = self._words_count(fps)
        tnps_count, _ = self._words_count(tnps)

        # fns_word_count, fns_word_num_total = self._filter_minimal_count(dict(fns_word_count_li))
        # fps_word_count, fps_word_num_total = self._filter_minimal_count(dict(fps_word_count_li))
        # tnps_word_count, tnps_word_num_total = self._filter_minimal_count(dict(tnps_word_count_li))

        fns_word_ratio = _fpp(dict(fns_count), self.word_count_dict, self.count_thresh)
        fps_word_ratio = _fpp(dict(fps_count), self.word_count_dict, self.count_thresh)
        tnps_word_ratio = _fpp(dict(tnps_count), self.word_count_dict, self.count_thresh)
        # sorted_fnp = sorted(fns_word_ratio.items(), key=lambda item: item[1])[::-1]
        sorted_fpp = sorted(fps_word_ratio.items(), key=lambda item: item[1])[::-1]

        if allow_change:
            new_suppress_words_ids = []
            cnt = 0
            for p in sorted_fpp:
                if p[1] <= self.filtering_thresh:
                    break

                cnt += 1
                if verbose:
                    logger.info('{:<12}: {:.3f}, [{:<3}, {:<5}]'.format(self.tokenizer.ids_to_tokens[p[0]], p[1], fps_count[p[0]], self.word_count_dict[p[0]]))

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
                self.word_appear_records.pop(w_ids)
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
        for w_ids in self.word_appear_records.keys():
            word_count_dict[w_ids] = sum(self.word_appear_records[w_ids])
        return word_count_dict

    def update_word_appear_records(self, w_ids, v):
        self.word_appear_records[w_ids].append(v)
        if len(self.word_appear_records[w_ids]) > self.window_size:
            self.word_appear_records[w_ids].pop(0)
    """
