from __future__ import absolute_import, division, print_function

import logging
import os
import random
import json
import pickle
import utils.utils as myUtils

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME, PHASE_NAMES
from bert.modeling import BertForSequenceClassification, BertConfig
from bert.tokenization import BertTokenizer
from bert.optimization import BertAdam, WarmupLinearSchedule

from loader import GabProcessor, WSProcessor, NytProcessor, convert_examples_to_features
import utils.config as config

# for hierarchical explanation algorithms
from hiex import SamplingAndOcclusionExplain


def unpack_features(fs, output_mode='classification'):
    input_ids = torch.tensor(np.array([f.input_ids for f in fs]), dtype=torch.long)
    input_mask = torch.tensor(np.array([f.input_mask for f in fs]), dtype=torch.long)
    segment_ids = torch.tensor(np.array([f.segment_ids for f in fs]), dtype=torch.long)
    if output_mode == 'regression':
        label_ids = torch.tensor(np.array([f.label_id for f in fs]), dtype=torch.float)
    else:
        label_ids = torch.tensor(np.array([f.label_id for f in fs]), dtype=torch.long)

    return input_ids, input_mask, segment_ids, label_ids


def get_dataloader(ds, args):
    data = TensorDataset(*ds)
    if args.local_rank == -1:
        sampler = RandomSampler(data)
    else:
        sampler = DistributedSampler(data)
    dl = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
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


def update_fpr_dict(attr_dict, stats_li, steps):
    fpr_dict, tnpr_dict, diff_dict = stats_li
    for w in attr_dict.keys():
        attr_obj = attr_dict[w]
        w_id = attr_obj.id
        fpr = fpr_dict[w_id] if w_id in fpr_dict else 0.
        tnpr = tnpr_dict[w_id] if w_id in tnpr_dict else 0.
        diff = diff_dict[w_id] if w_id in diff_dict else 0.
        attr_obj.fpr_changes[steps] = [fpr, tnpr, diff]


class PowerOfExplanation:
    def __init__(self, args, device):
        self.args = args
        self.supported_modes = ['vanilla', 'mid', 'soc']
        self._mode = 'mid'
        self.device = device
        self.phases, self.phases_iter, self.phase = [], None, -1

        self.global_step, self.step_in_phase, self.num_labels = 0, 0, 0
        self.losses, self.reg_losses = [], []
        self.suppress_records = [[], []]
        self.attr_change_dict, self.manual_change_dict = dict(), dict()     # dict of AttrRecord

        self.explainer, self.model, self.optimizer = None, None, None
        self.logger, self.tokenizer, self.processor = None, None, None
        self.ds_train, self.ds_eval = None, None    # ids, mask, segment_ids, label
        self.dl_train, self.dl_eval = None, None

        class_weight = torch.FloatTensor([self.args.negative_weight, 1]).to(self.device)
        self.loss_fct = CrossEntropyLoss(class_weight)  # FIXME: support different losses

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

    def set_training_mode(self, mode):
        assert mode in self.supported_modes, 'Unknown mode: [{}], currently support: {}'.format(mode, self.supported_modes)
        self._mode = mode
        # Set iterators
        if mode == 'vanilla':
            self.phases = [0]
        elif mode == 'soc':
            self.phases = [1]
        else:
            self.phases = [0, 1, 2]
        self.phases_iter = iter(self.phases)

    def init_trainer(self):
        # if self.args.get_attr:
        #     init_manual_attr_dict()
        # TODO: initialize if needed manual list

        self.dl_train = get_dataloader(self.ds_train, self.args)
        self.dl_eval = get_dataloader(self.ds_eval, self.args)
        pass

    def train(self):
        self.phase = next(self.phases_iter, -1)
        no_progress_cnt = 0
        # add output_mode=='regression' in case needed

        for epoch in range(int(self.args.num_train_epochs)):
            self.logger.info('***** Epoch {} *****'.format(epoch))
            tr_loss = 0
            for step, batch in enumerate(tqdm(self.dl_train, desc='Batches')):
                if self.phase < 0:
                    break
                ''' input_ids, mask, segment_ids, label '''
                batch = tuple(t.to(self.device) for t in batch)

                ''' ==================================================== '''
                ''' |                 Cross entropy loss               | '''
                ''' ==================================================== '''
                # define a new function to compute loss values for both output_modes
                logits = self.model(batch[0], batch[2], batch[1])   # be careful with the order

                # Update loss in case n_gpu/gradient_accumulation is set
                # TODO: check this, defined before actual training starts
                loss = self.loss_fct(logits.view(-1, self.num_labels), batch[-1].view(-1))

                tr_loss += loss.item()
                loss.backward()

                ''' ==================================================== '''
                ''' |                 Regularization term              | '''
                ''' ==================================================== '''
                if self.phase == 1:
                    # Note that the backpropagation happens within the function
                    # TODO: different suppression strategies
                    reg_loss, _ = self.explainer.suppress_explanation_loss(*batch, do_backprop=True)
                else:
                    # Output reg_loss only for recording
                    reg_loss = 0.
                self.losses.append(loss.item())
                self.reg_losses.append(loss.item() + reg_loss)

                if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:     # Batch normalization
                    # FIXME: support FP16 if needed
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.global_step += 1
                self.step_in_phase += 1

                if self.global_step % self.args.reg_steps == 0:
                    self.logger.info('***** Update attribution records at #{} *****'.format(self.global_step))
                    self.update_fpp_window(allow_change=(self.phase == 0))
                    self.validate(use_train=self.args.attr_on_training)    # TODO: add the argument
                    # self.update_attr_dict()

    def update_attr_dict(self):
        new_ws = list(set(self.explainer.get_suppress_words()) - set(self.suppress_records[-1]))
        if self.args.get_attr:
            # TODO: check whether the actual address is given, experiment
            tmp_change_dict = self.attr_change_dict
        else:
            tmp_change_dict = dict()

        for w in new_ws:
            tmp_change_dict[w] = myUtils.AttrRecord(w, self.tokenizer.vocab[w])
        return tmp_change_dict

    def update_fpp_window(self, allow_change, use_train=1, verbose=0):
        if not allow_change and not self.args.get_attr:
            return  # return if not need to update or record
        self.model.train(False)
        eval_dl = self.dl_train if use_train else self.dl_eval

        self.logger.info('\t--> Updating suppressing list')
        self.logger.info('\t\tNum examples = %d', len(eval_dl.dataset))
        wrong_li = [[] for _ in range(4)]
        right_li = [[] for _ in range(4)]
        for i, batch in enumerate(eval_dl):
            idxs_wrong = find_incorrect(self.model, batch, self.device)
            idxs_all = list(range(len(batch[0])))
            for j in range(len(wrong_li)):
                wrong_li[j] += [batch[j][idx] for idx in idxs_wrong]
                right_li[j] += [batch[j][idx] for idx in idxs_all if idx not in idxs_wrong]
        
        if self.args.suppress_lazy:
            # the most updated version
            # TODO: separate the process of extracting the FPR
            stats_li = self.explainer.update_suppress_words_lazy(
                [wrong_li[0], wrong_li[-1]], [right_li[0], right_li[-1]],
                verbose=0, allow_change=allow_change)
            if self.args.get_attr:
                self.update_fpr(stats_li)
        else:
            self.explainer.update_suppress_words(
                [wrong_li[0], wrong_li[-1]], [right_li[0], right_li[-1]],
                verbose=verbose)

        # self.suppress_records.append(self.explainer.get_suppress_words())     # this should be done after checking att
        self.model.train(True)

    def update_fpr(self, stats_li):
        update_fpr_dict(self.manual_change_dict, stats_li, self.global_step)
        update_fpr_dict(self.attr_change_dict, stats_li, self.global_step)

    def _update_changes_dict(self, attr_dict, inputs):
        for w in attr_dict:
            obj = attr_dict[w]
            if not obj.check_test:  # record the instances containing target word to save time from checking
                myUtils.find_positions(inputs[0], obj)
            myUtils.record_attr_change(self.explainer, inputs[:3], obj, self.global_step)

    def update_changes_dict(self, attr_dict, inputs):
        all_ids = inputs[0]
        self.logger.info(attr_dict.keys())
        self._update_changes_dict(attr_dict, inputs)
        if self.args.get_attr:
            self._update_changes_dict(self.manual_change_dict, inputs)

    def validate(self, use_train=0):
        """
        Validate both the model & the suspicious words
        """
        eval_dl = self.dl_train if use_train else self.dl_eval
        self.logger.info('\t--> Running evaluation')
        self.logger.info('\t\tNum examples = %d', len(eval_dl.dataset()))

        self.model.train(False)

        attr_dict = self.update_attr_dict()
        self.update_changes_dict(attr_dict, self.ds_train if use_train else self.ds_eval)

        eval_loss, eval_loss_reg = 0, 0
        eval_reg_cnt, nb_eval_steps = 0, 0
        preds = []
        # for detailed prediction results
        input_seqs = []

        # for input_ids, input_mask, segment_ids, label_ids in eval_dl:
        for step, batch in enumerate(tqdm(eval_dl, desc='Validate')):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                logits = self.model(batch[0], batch[2], batch[1], labels=None)   # be careful with the order
            tmp_eval_loss = self.loss_fct(logits.view(-1, self.num_labels), batch[-1].view(-1))
            eval_loss += tmp_eval_loss.mean().item()

            if self.args.reg_explanations:
                with torch.no_grad():
                    # TODO: handle different situation by the explainer in the future
                    if self.args.neutral_words_file != '':
                        reg_loss, reg_cnt = self.explainer.compute_explanation_loss(*batch, backprop=False)
                    else:
                        reg_loss, reg_cnt = self.explainer.suppress_explanation_loss(*batch, backprop=False)
                eval_loss_reg += reg_loss
                eval_reg_cnt += reg_cnt
            # TODO: Solving the bugs
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

            for b in range(input_ids.size(0)):
                i = 0
                while i < input_ids.size(1) and input_ids[b, i].item() != 0:
                    i += 1
                token_list = tokenizer.convert_ids_to_tokens(input_ids[b, :i].cpu().numpy().tolist())
                input_seqs.append(' '.join(token_list))

        eval_loss = eval_loss / nb_eval_steps
        eval_loss_reg = eval_loss_reg / (eval_reg_cnt + 1e-10)
        preds = preds[0]
        if output_mode == "classification":
            pred_labels = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            pred_labels = np.squeeze(preds)
        pred_prob = F.softmax(torch.from_numpy(preds).float(), -1).numpy()
        result = compute_metrics(task_name, pred_labels, all_label_ids.numpy(), pred_prob)
        loss = tr_loss / (global_step + 1e-10) if args.do_train else None

        result['eval_loss'] = eval_loss
        result['eval_loss_reg'] = eval_loss_reg
        result['global_step'] = global_step
        result['loss'] = loss

        split = 'dev' if not args.test else 'test'

        output_eval_file = os.path.join(args.output_dir, "eval_results_%d_%s_%s.txt"
                                        % (global_step, split, args.task_name))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("Epoch %d" % epoch)
            for key in sorted(result.keys()):
                if type(result[key]) is np.float64:
                    logger.info("  {} = {:.4f}".format(key, result[key]))
                else:
                    logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        output_detail_file = os.path.join(args.output_dir, "eval_details_%d_%s_%s.txt"
                                          % (global_step, split, args.task_name))
        with open(output_detail_file, 'w') as writer:
            for i, seq in enumerate(input_seqs):
                pred = preds[i]
                gt = all_label_ids[i]
                writer.write('{}\t{}\t{}\n'.format(gt, pred, seq))

        model.train(True)
        return result

    def test(self):     # separate test on testing set from validation set
        # TODO: loading data first
        pass
