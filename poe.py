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


def unpack_features(fs, args, output_mode='classification'):
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
    matched = matched.to('cpu')
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
        self.device = device
        self.phases = []
        self.phases_iter = None
        self.phase = -1

        self.num_labels = 0
        self.global_step, self.step_in_phase = 0, 0
        self.losses, self.reg_losses = [], []
        self.suppress_records = [[], []]
        self.attr_change_dict, self.manual_change_dict = dict(), dict()

        self.model, self.optimizer = None, None
        self.explainer = None
        self.logger, self.tokenizer, self.processor = None, None, None
        self.ds_train, self.ds_eval = None, None    # ids, mask, segment_ids, label
        self.dl_train, self.dl_eval = None, None

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
            # init_manual_attr_dict()
        # TODO: initialize if needed manual list
        pass

    def train(self):
        self.phase = next(self.phases_iter, -1)
        no_progress_cnt = 0
        class_weight = torch.FloatTensor([self.args.negative_weight, 1]).to(self.device)
        loss_fct = CrossEntropyLoss(class_weight)
        # add output_mode=='regression' in case needed

        self.dl_train = get_dataloader(self.ds_train, self.args)
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
                logits = self.model(batch[0], batch[2], batch[1])   # be care with the order

                # Update loss in case n_gpu/gradient_accumulation is set
                loss = loss_fct(logits.view(-1, self.num_labels), batch[-1].view(-1))

                tr_loss += loss.item()
                loss.backward()

                ''' ==================================================== '''
                ''' |                 Regularization term              | '''
                ''' ==================================================== '''
                if self.phase == 1:
                    reg_loss, _ = self.explainer.suppress_explanation_loss(*batch, do_backprop=True)
                else:
                    reg_loss = 0.
                self.losses.append(loss.item())
                self.reg_losses.append(loss.item() + reg_loss)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                self.step_in_phase += 1

                if self.global_step % self.args.reg_steps == 0:
                    self.logger.info('***** Update attribution records at #{} *****'.format(self.global_step))
                    self.update_sup_list(allow_change=(self.phase == 0))
                    self.update_attr_dict()
                    # TODO: validation

    def update_attr_dict(self):
        new_ws = list(set(self.suppress_records[-1]) - set(self.suppress_records[-2]))
        for w in new_ws:
            self.attr_change_dict[w] = myUtils.AttrRecord(w, self.tokenizer.vocab[w])

    def update_sup_list(self, allow_change, use_train=1):
        if not allow_change and not self.args.get_attr:
            return  # return if not need to update or record
        self.model.train(False)
        if use_train:
            eval_dl = self.dl_train
        else:
            eval_dl = self.dl_eval

        self.logger.info('-\tUpdating suppressing list')
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
            stats_li = self.explainer.update_suppress_words_lazy([wrong_li[0], wrong_li[-1]], [right_li[0], right_li[-1]], verbose=0, allow_change=allow_change)
            if self.args.get_attr:
                self.update_fpr(stats_li)
        else:
            self.explainer.update_suppress_words([wrong_li[0], wrong_li[-1]], [right_li[0], right_li[-1]], verbose=verbose)

        self.suppress_records.append(self.explainer.get_suppress_words())
        self.model.train(True)

    def update_fpr(self, stats_li):
        update_fpr_dict(self.manual_change_dict, stats_li, self.global_step)
        update_fpr_dict(self.attr_change_dict, stats_li, self.global_step)
