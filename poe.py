from __future__ import absolute_import, division, print_function

import os
import utils.utils as my_utils
from tqdm import tqdm
import time
import pickle
import string

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch import nn

# from bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME, PHASE_NAMES
# from bert.modeling import BertForSequenceClassification, BertConfig
# from bert.tokenization import BertTokenizer
# from bert.optimization import BertAdam, WarmupLinearSchedule
# from loader import GabProcessor, WSProcessor, NytProcessor, convert_examples_to_features
# from hiex import SamplingAndOcclusionExplain

VERSION = 'Current version of MiD: 1.001.001'


def unpack_features(fs, output_mode='classification'):
    input_ids = torch.tensor(np.array([f.input_ids for f in fs]), dtype=torch.long)
    input_mask = torch.tensor(np.array([f.input_mask for f in fs]), dtype=torch.long)
    segment_ids = torch.tensor(np.array([f.segment_ids for f in fs]), dtype=torch.long)
    if output_mode == 'regression':
        label_ids = torch.tensor(np.array([f.label_id for f in fs]), dtype=torch.float)
    else:
        label_ids = torch.tensor(np.array([f.label_id for f in fs]), dtype=torch.long)

    return input_ids, input_mask, segment_ids, label_ids


def get_dataloader(ds, args, train=True):
    data = TensorDataset(*ds)
    if args.local_rank == -1:
        sampler = RandomSampler(data)
    else:
        sampler = DistributedSampler(data)
    # if not train:
    #     sampler = SequentialSampler(data)
    size = args.train_batch_size if train else args.eval_batch_size
    dl = DataLoader(data, sampler=sampler, batch_size=size)
    return dl


def find_incorrect(model, batch, device='cuda'):
    # make sure you have set train to False cause this function won't do that
    # model.train(False)
    input_ids = batch[0].to(device)
    input_mask = batch[1].to(device)
    segment_ids = None  # Not used
    label_ids = batch[3].to(device)

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)
        pred = logits.max(1).indices
    matched = pred == label_ids
    matched = matched.to('cpu').numpy()
    return np.where(matched == 0)[0]


def feature_fpp(count_dict, total_dict, thresh):
    dict_ratio = dict()
    for w in count_dict.keys():
        if total_dict[w] < thresh:
            continue
        dict_ratio[w] = float(count_dict[w]) / total_dict[w]
    return dict_ratio


def update_fpp_dict(attr_dict, stats_li, steps):
    fpp_dict, tnpp_dict, fnp_dict = stats_li
    for w in attr_dict.keys():
        attr_obj = attr_dict[w]
        w_id = attr_obj.id
        fpp = fpp_dict[w_id] if w_id in fpp_dict else 0.
        tnpp = tnpp_dict[w_id] if w_id in tnpp_dict else 0.
        fnp = fnp_dict[w_id] if w_id in fnp_dict else 0.
        attr_obj.fpr_changes[steps] = [fpp, tnpp, fnp]


class MiD:
    def __init__(self, args, device, logger, tokenizer, processor):
        logger.info(my_utils.heading(VERSION))
        self.args = args
        self.supported_modes = ['vanilla', 'mid', 'soc']
        self.phases, self.phases_iter, self.phase = [], None, -1
        self._mode = self.set_training_mode()
        self.device = device

        ''' For training and recording '''
        self.logger, self.tokenizer, self.processor = logger, tokenizer, processor
        self.global_step, self.step_in_phase, self.num_labels = 0, 0, len(self.processor.get_labels())
        self.losses, self.reg_losses = [], []
        self.suppress_records = [[], []]
        self.attr_change_dict, self.manual_change_dict = dict(), dict()  # dict of AttrRecord

        self.explainer, self.model, self.optimizer = None, None, None
        self.ds_train, self.ds_eval = None, None  # ids, mask, segment_ids, label
        self.dl_train, self.dl_eval = None, None

        class_weight = torch.FloatTensor([self.args.negative_weight, 1]).to(self.device)
        self.loss_fct = nn.CrossEntropyLoss(class_weight)
        self.desc, self.pbar = my_utils.DescStr(), None  # For nested progress bar

        ''' Sliding window for word detector '''
        self.neutral_words, self.neutral_words_ids = my_utils.load_suppress_words(args.neutral_words_file, self.tokenizer, args.suppress_weighted)
        self.neg_suppress_words, self.neg_suppress_words_ids = my_utils.load_suppress_words('', self.tokenizer, args.suppress_weighted)
        self.pos_suppress_words, self.pos_suppress_words_ids = my_utils.load_suppress_words('', self.tokenizer, args.suppress_weighted)

        self.stop_words, self.stop_words_ids = self.get_stop_words()
        self.word_count_dict, self.word_appear_records = dict(), dict()
        self.window_count = args.window_size * args.ratio_in_window

    def load_tools(self, tokenizer, processor, logger):
        self.logger = logger
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_labels = len(self.processor.get_labels())

    def load_model(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def load_explainer(self, explainer):
        self.explainer = explainer

    def load_data(self, train_features, eval_features):
        self.ds_train = unpack_features(train_features, self.args)
        self.ds_eval = unpack_features(eval_features, self.args)
        self.get_global_words_count(np.array([f.input_ids for f in train_features], dtype=np.int64))

    def set_training_mode(self):
        mode = self.args.mode
        assert mode in self.supported_modes, 'Unknown mode: [{}], only support: {}'.format(mode, self.supported_modes)
        # Set iterators
        if mode == 'vanilla':
            self.phases = [0]
        elif mode == 'soc':
            self.phases = [1]
        else:
            self.phases = [0, 1, 2]
        self.phases_iter = iter(self.phases)
        return mode

    def init_trainer(self):
        """
        Init the trainer just before the training, such that no information will be omitted
        """
        if self.args.get_attr or self._mode:
            self.init_manual_list()

        self.dl_train = get_dataloader(self.ds_train, self.args)
        self.dl_eval = get_dataloader(self.ds_eval, self.args, train=False)

    def init_manual_list(self):
        ws = self.neutral_words
        self.manual_change_dict = dict()
        for w in ws:
            self.manual_change_dict[w] = my_utils.AttrRecord(w, self.tokenizer.vocab[w])

    def train(self):
        self.init_trainer()     # TODO: 6. handling the case of resuming training

        start_at = time.time()
        self.phase = next(self.phases_iter, -1)
        no_progress_cnt, val_best_f1 = 0, -1
        val_best_results, val_phase_names = [], []
        val_best = None

        with tqdm(total=len(self.phases)*self.args.max_iter, ncols=my_utils.MAX_LINE_WIDTH, desc='#Iter') as pbar:
            self.pbar = pbar
            while self.phase >= 0:
                tr_loss = 0
                # for step, batch in enumerate(tqdm(self.dl_train, desc='Batches')):
                for step, batch in enumerate(self.dl_train):
                    if self.phase < 0:
                        break
                    ''' input_ids, mask, segment_ids, label '''
                    batch = tuple(t.to(self.device) for t in batch)

                    ''' ==================================================== 
                        |                 Cross entropy loss               | 
                        ==================================================== '''
                    # define a new function to compute loss values for both output_modes
                    logits = self.model(batch[0], batch[2], batch[1])  # be careful with the order

                    # Update loss in case n_gpu/gradient_accumulation is set
                    loss = self.loss_fct(logits.view(-1, self.num_labels), batch[-1].view(-1))

                    tr_loss += loss.item()
                    loss.backward()

                    ''' ==================================================== 
                        |                 Regularization term              | 
                        ==================================================== '''
                    if self.phase == 1:
                        # Note that the backpropagation happens within the function
                        targets = self.neg_suppress_words_ids if self._mode == 'mid' else self.neutral_words_ids
                        ignore = 0 if self._mode == 'mid' else -1
                        reg_loss, _ = self.explainer.suppress_explanation_loss(batch, targets, ignore_label=ignore, do_backprop=True)
                        # if self._mode == 'mid':
                        #     reg_loss, _ = self.explainer.suppress_explanation_loss(*batch, do_backprop=True)
                        # else:
                        #     reg_loss, _ = self.explainer.compute_explanation_loss(*batch, do_backprop=True)
                    else:
                        reg_loss = 0.
                    self.losses.append(loss.item())
                    self.reg_losses.append(loss.item() + reg_loss)

                    if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:  # Batch normalization
                        # FIXME: support FP16 if needed
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    self.global_step += 1
                    self.step_in_phase += 1
                    self.pbar.update(1)

                    if self.global_step % self.args.reg_steps == 0:
                        # Update suppressing list, and record the current best version if the constraint satisfies
                        ''' ==================================================== 
                            |          Maintaining the suppression list        | 
                            ==================================================== '''
                        self.logger.info('\n')
                        self.logger.info('***** Update attribution records at #{} *****'.format(self.global_step))
                        self.update_fpp_window(allow_change=(self.phase == 0))
                        val_res = self.validate(tr_loss, use_train=self.args.attr_on_training)

                        ''' ==================================================== 
                            |            Recording the best version            |
                            ==================================================== '''
                        val_acc, val_f1 = val_res['acc'], val_res['f1']
                        if self.global_step % self.args.val_steps == 0:
                            self.logger.info("***** Validation *****")
                            if val_f1 > val_best_f1:
                                val_best_f1 = val_f1
                                val_best = val_res
                                if self.args.local_rank == -1 or torch.distributed.get_rank() == 0:
                                    my_utils.save_model(self.args, self.model, self.tokenizer,
                                                        phase=self.phase, postfix='_best')
                            else:
                                # halve the learning rate
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] *= 0.5
                                no_progress_cnt += 1
                                self.logger.info("--> Reducing learning rate...")
                                self.logger.info('--> No progress count: {}'.format(no_progress_cnt))

                            lr_str = ''
                            for param_group in self.optimizer.param_groups:
                                lr_str += str(param_group['lr']) + '  '
                            self.logger.info("Current learning rate: {}".format(lr_str))

                        ''' ==================================================== 
                            |           Switching to the next phase            | 
                            ==================================================== '''
                        if self.step_in_phase >= self.args.max_iter > 0:
                            # Exit phase when criterion satisfies
                            # TODO: 5. save status everytime a phase ends, for resuming training after disconnection
                            val_best_results.append(val_best)  # Only for recording
                            val_phase_names.append(my_utils.PHASE_NAMES[self.phase])
                            my_utils.save_model(self.args, self.model, self.tokenizer, phase=self.phase, postfix='_final')
                            self.phase = next(self.phases_iter, -1)
                            self.logger.info('-' * my_utils.MAX_LINE_WIDTH)
                            # self.logger.info('|            Move to next phase: {}              |'.format(self.phase))
                            self.logger.info(my_utils.heading('Move to next phase: {}'.format(self.phase)))
                            self.logger.info('-' * my_utils.MAX_LINE_WIDTH)

                            # make sure that model from new phase will be recorded
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.args.learning_rate / 2.
                            val_best_f1 = -1
                            no_progress_cnt = 0
                            self.step_in_phase = 0
                            if self.phase == 2:
                                self.args.max_iter += self.args.extra_iter
        self.logger.info('\n')
        self.logger.info('--> Training complete')
        time_cost = my_utils.seconds2hms(time.time() - start_at)

        for i, name in enumerate(val_phase_names):
            self.logger.info('Best performing model on [{}]'.format(name))
            result = val_best_results[i]
            for key in sorted(result.keys()):
                self.logger.info("\t{} = {:.4f}".format(key, result[key]))
        self.logger.info('\n')
        self.logger.info('--> Time cost: {}:{}:{}\n'.format(*time_cost))

        suppress_records = self.suppress_records[2:]
        records = [self.losses, self.reg_losses, suppress_records, self.attr_change_dict, self.manual_change_dict]
        with open(self.args.stats_file, 'wb') as f:
            pickle.dump(records, f)
            self.logger.info('Save to {}'.format(self.args.stats_file))

    def update_attr_dict(self):
        new_ws = list(set(self._get_suppress_words()) - set(self.suppress_records[-1]))
        if self.args.get_attr:
            tmp_change_dict = self.attr_change_dict
        else:
            tmp_change_dict = dict()

        for w in new_ws:
            if w in tmp_change_dict: self.logger.warning('Old list already contain new term [{}]'.format(w))
            tmp_change_dict[w] = my_utils.AttrRecord(w, self.tokenizer.vocab[w])
        return tmp_change_dict, new_ws

    def update_fpp_window(self, allow_change, use_train=1, verbose=0):
        if not allow_change and not self.args.get_attr:
            return  # return if not need to update or record
        self.model.train(False)
        eval_dl = self.dl_train if use_train else self.dl_eval

        self.logger.info('\t--> Updating suppressing list through FPP')
        self.logger.info('\t\tNum examples = %d', len(eval_dl.dataset))
        wrong_li = [[] for _ in range(4)]
        right_li = [[] for _ in range(4)]
        for i, batch in enumerate(tqdm(eval_dl, file=self.desc, desc='FPP')):
            idxs_wrong = find_incorrect(self.model, batch, self.device)
            idxs_all = list(range(len(batch[0])))
            for j in range(len(wrong_li)):
                wrong_li[j] += [batch[j][idx] for idx in idxs_wrong]
                right_li[j] += [batch[j][idx] for idx in idxs_all if idx not in idxs_wrong]
            self.pbar.set_description(self.desc.read())

        if self.args.mode == 'mid':
            # the most updated version
            stats_li = self.update_suppress_words(wrong_li, right_li, allow_change=allow_change, verbose=verbose)
            if self.args.get_attr:
                self.update_fpp(stats_li)

        self.model.train(True)

    def update_fpp(self, stats_li):
        update_fpp_dict(self.manual_change_dict, stats_li, self.global_step)
        update_fpp_dict(self.attr_change_dict, stats_li, self.global_step)

    def _update_changes_dict(self, attr_dict, inputs):
        for w in attr_dict:
            obj = attr_dict[w]
            if not obj.checked_test:  # record the instances containing target word to save time from checking
                my_utils.find_positions(inputs[0], obj)
            my_utils.record_attr_change(self.explainer, inputs[:3], obj, self.global_step)

    def update_changes_dict(self, attr_dict, inputs):
        self.model.train(False)
        self.logger.info('\t\t{}'.format(attr_dict.keys()))
        self._update_changes_dict(attr_dict, inputs)
        if self.args.get_attr:
            self._update_changes_dict(self.manual_change_dict, inputs)
        self.model.train(True)

    def update_suppressing_list(self, attr_dict, new_ws):
        """
        Update the suppressing list only during vanilla stage under 'mid' mode
        """
        if self._mode == 'mid' and self.phase == 0:
            filtering = set()
            for w in new_ws:
                avg_attr = np.mean(next(iter(attr_dict[w].attr_changes.values())))
                if abs(avg_attr) < self.args.tau:
                    filtering.add(w)
                    del attr_dict[w]
                    self.logger.info('\t\tFiltered word: {:>15} with avg. attr: {:.3f}'.format(w, avg_attr))

            self.filter_suppress_words(filtering)
            self.logger.info('\t\tFinal check with words that are removed:')
            removed = set(new_ws) - set(self._get_suppress_words())
            self.logger.info('\t\t{}'.format(removed))
            self.logger.info('\t\t' + '-'*20)

            self.logger.info('\t\t------- Current Suppressing List --------')
            self.logger.info('\t\t{}'.format(self.neg_suppress_words))

            self.suppress_records.append(self._get_suppress_words())

    def validate(self, tr_loss, use_train=0):
        """
        Validate both the model & the suspicious words
        """
        eval_dl = self.dl_train if use_train else self.dl_eval
        eval_ds = self.ds_train if use_train else self.ds_eval

        attr_dict, new_ws = self.update_attr_dict()

        self.logger.info('\n')
        self.logger.info('\t--> Verify candidates through SOC')
        self.update_changes_dict(attr_dict, eval_ds)
        self.update_suppressing_list(attr_dict, new_ws)

        self.logger.info('\t--> Running evaluation')
        self.logger.info('\t\tNum examples = %d', len(eval_dl.dataset))
        return self._validate(eval_dl, tr_loss)

    def _validate(self, dl, tr_loss=0.):
        self.model.train(False)
        eval_loss, eval_loss_reg = 0, 0
        eval_reg_cnt, nb_eval_steps = 0, 0
        preds, ys = [], []
        # for detailed prediction results
        input_seqs = []
        loss_fct = nn.CrossEntropyLoss()

        # for step, batch in enumerate(dl):
        for step, batch in enumerate(tqdm(dl, file=self.desc, desc='#Eval')):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                logits = self.model(batch[0], batch[2], batch[1], labels=None)  # be careful with the order
            tmp_eval_loss = loss_fct(logits.view(-1, self.num_labels), batch[-1].view(-1))
            eval_loss += tmp_eval_loss.mean().item()

            # if self.args.reg_explanations:
            with torch.no_grad():
                if self._mode == 'mid' or self._mode == 'soc':
                    targets = self.neg_suppress_words_ids if self._mode == 'mid' else self.neutral_words_ids
                    ignore = 0 if self._mode == 'mid' else -1
                    reg_loss, reg_cnt = self.explainer.suppress_explanation_loss(batch, targets, ignore_label=ignore, do_backprop=False)
                else:
                    reg_loss, reg_cnt = 0, 0
                # if self._mode == 'mid':
                #     reg_loss, reg_cnt = self.explainer.suppress_explanation_loss(*batch, do_backprop=False)
                # elif self._mode == 'soc':
                #     reg_loss, reg_cnt = self.explainer.compute_explanation_loss(*batch, do_backprop=False)
                # else:
                #     reg_loss, reg_cnt = 0, 0
            eval_loss_reg += reg_loss
            eval_reg_cnt += reg_cnt
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                ys.append(batch[-1].detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                ys[0] = np.append(ys[0], batch[-1].detach().cpu().numpy(), axis=0)

            for b in range(batch[0].size(0)):
                i = 0
                while i < batch[0].size(1) and batch[0][b, i].item() != 0:
                    i += 1
                token_list = self.tokenizer.convert_ids_to_tokens(batch[0][b, :i].cpu().numpy().tolist())
                input_seqs.append(' '.join(token_list))

            self.pbar.set_description(self.desc.read())

        eval_loss = eval_loss / nb_eval_steps
        eval_loss_reg = eval_loss_reg / (eval_reg_cnt + 1e-10)
        preds = preds[0]
        ys = ys[0]
        pred_labels = np.argmax(preds, axis=1)  # FIXME: Currently, only support classification
        pred_prob = nn.functional.softmax(torch.from_numpy(preds).float(), -1).numpy()
        # result = my_utils.compute_metrics(pred_labels, all_label_ids.numpy(), pred_prob)
        result = my_utils.compute_metrics(pred_labels, ys, pred_prob)
        loss = tr_loss / (self.global_step + 1e-10) if self.args.do_train else None

        result['eval_loss'] = eval_loss
        result['eval_loss_reg'] = eval_loss_reg
        result['loss'] = loss

        split = 'dev'
        output_eval_file = os.path.join(self.args.output_dir, "eval_results_%d_%s_%s.txt"
                                        % (self.global_step, split, self.args.task_name))
        with open(output_eval_file, "w") as writer:
            self.logger.info('\n')
            self.logger.info("\t***** Eval results *****")
            self.logger.info("\t\tGlobal steps {}".format(self.global_step))
            for key in sorted(result.keys()):
                self.logger.info("\t\t{} = {:.4f}".format(key, result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write('{} = {}\n'.format('global_step', self.global_step))

        output_detail_file = os.path.join(self.args.output_dir, "eval_details_%d_%s_%s.txt"
                                          % (self.global_step, split, self.args.task_name))
        with open(output_detail_file, 'w') as writer:
            for i, seq in enumerate(input_seqs):
                pred = preds[i]
                gt = ys[i]
                writer.write('{}\t{}\t{}\n'.format(gt, pred, seq))

        self.model.train(True)
        return result

    """
    ================================================================
    |                  Suspicious detector                         |
    ================================================================
    """
    def _get_suppress_words(self):
        return self.neg_suppress_words.copy()

    def get_global_words_count(self, corpus):
        counter, _ = my_utils.words_count(corpus, self.stop_words_ids)
        for wid, cnt in counter:
            self.word_count_dict[wid] = cnt

    def get_ratios(self, group_li):
        res = []
        for group in group_li:
            count, _ = my_utils.words_count(group, self.stop_words_ids)
            res.append(feature_fpp(dict(count), self.word_count_dict, self.args.count_thresh))
        return res

    def update_suppress_words(self, wrong_li, right_li, verbose=0, allow_change=False):
        fns, fps, tns, tps = [], [], [], []
        for idx in range(len(wrong_li[0])):
            if wrong_li[-1][idx] == 1:
                # the ground truth is positive, indicating the instance is false negative instance
                fns.append(wrong_li[0][idx].numpy().tolist())
            else:
                fps.append(wrong_li[0][idx].numpy().tolist())

        for idx in range(len(right_li[0])):
            if right_li[-1][idx] == 1:
                tps.append(right_li[0][idx].numpy().tolist())
            else:
                tns.append(right_li[0][idx].numpy().tolist())

        tnps = tns + tps
        fns_word_ratio, fps_word_ratio, tnps_word_ratio = self.get_ratios([fns, fps, tnps])
        # sorted_fnp = sorted(fns_word_ratio.items(), key=lambda item: item[1])[::-1]
        sorted_fpp = sorted(fps_word_ratio.items(), key=lambda item: item[1])[::-1]

        if allow_change:
            new_suppress_words_ids = []
            cnt = 0
            for p in sorted_fpp:
                if p[1] <= self.args.eta:
                    break

                cnt += 1
                if verbose:
                    self.logger.info('{:<12}: {:.3f} [{:<5}]'.format(self.tokenizer.ids_to_tokens[p[0]], p[1], self.word_count_dict[p[0]]))

                target = p[0]
                new_suppress_words_ids.append(target)
                if target not in self.word_appear_records:
                    self.word_appear_records[target] = [1]
                else:
                    self.update_word_appear_records(target, 1)

            for w_ids in self.word_appear_records.keys():
                if w_ids not in new_suppress_words_ids:
                    self.update_word_appear_records(w_ids, 0)
            self._update_suppress_words()
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
        if len(self.word_appear_records[w_ids]) > self.args.window_size:
            self.word_appear_records[w_ids].pop(0)

    def get_stop_words(self):
        stop_words = {'[CLS]', '[PAD]', '[SEP]'}
        for pc in string.punctuation:
            stop_words.add(pc)

        stop_words_ids = set()
        for s in stop_words:
            try:
                stop_words_ids.add(self.tokenizer.vocab[s])
            except KeyError:
                self.logger.warning('=' * my_utils.MAX_LINE_WIDTH)
                self.logger.warning('WARNING: Cannot find target word in vocab:\t{}'.format(s))
                self.logger.warning('=' * my_utils.MAX_LINE_WIDTH + '\n\n')
        return stop_words, stop_words_ids
