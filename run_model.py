# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Running BERT finetuning & evaluation on hate speech classification datasets.

Integrated with SOC explanation regularization

"""

from __future__ import absolute_import, division, print_function

import logging
import os
import random
import json
import pickle

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

from bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from bert.modeling import BertForSequenceClassification, BertConfig
from bert.tokenization import BertTokenizer
from bert.optimization import BertAdam, WarmupLinearSchedule

from loader import GabProcessor, WSProcessor, NytProcessor, convert_examples_to_features
import utils.config as config

# for hierarchical explanation algorithms
from hiex import SamplingAndOcclusionExplain

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, pred_probs):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    p, r = precision_score(y_true=labels, y_pred=preds), recall_score(y_true=labels, y_pred=preds)
    try:
        roc = roc_auc_score(y_true=labels, y_score=pred_probs[:,1])
    except ValueError:
        roc = 0.
    return {
        "acc": acc,
        "f1": f1,
        "precision": p,
        "recall": r,
        "auc_roc": roc
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels, pred_probs):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels, pred_probs)


def find_no_matching(logits, labels):
    pred = logits.detach().cpu().max(1).indices
    matched = pred == labels
    return np.where(matched == 0)[0]


def find_incorrect_pred(model, input_batch, device='cuda'):
    # make sure you have set train to False cause this funciton won't do that
    input_ids = input_batch[0].to(device)
    segment_ids = None
    input_mask = input_batch[1].to(device)
    label_ids = input_batch[3].to(device)

    model.train(False)
    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)
        pred = logits.max(1).indices
    matched = pred == label_ids
    matched = matched.to('cpu')
    return np.where(matched == 0)[0]


def main():
    logger.info('Current version of run_model: 0.000.1')

    parser = config.get_new_parser()
    parser = config.add_general_configs(parser)
    parser = config.add_explanation_configs(parser)
    parser = config.add_training_configs(parser)
    args = parser.parse_args()

    config.combine_args(args, config.configs)
    # args = configs

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        'gab': GabProcessor,
        'ws': WSProcessor,
        'nyt': NytProcessor
    }

    output_modes = {
        'gab': 'classification',
        'ws': 'classification',
        'nyt': 'classification'
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # save configs
    if args.do_train:
        f = open(os.path.join(args.output_dir, 'args.json'), 'w')
        json.dump(args.__dict__, f, indent=4)
        f.close()

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    processor = processors[task_name](args, tokenizer=tokenizer)
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    if args.do_train:
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                              cache_dir=cache_dir,
                                                              num_labels=num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
    model.to(device)

    if args.fp16:
        model.half()

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        if args.do_train:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss, tr_reg_loss = 0, 0
    tr_reg_cnt = 0
    epoch = -1
    val_best_f1 = -1
    val_best_loss = 1e10
    if args.max_iter < 0:
        early_stop_countdown = args.early_stop
    else:
        early_stop_countdown = int(args.max_iter / 200) + 1

    logger.info('***** Explanation config *****')
    logger.info('\tReg explanation:  \t{}'.format(args.reg_explanations))
    logger.info('\tReg balanced:     \t{}'.format(args.reg_balanced))
    logger.info('\tWeighted suppress:\t{}'.format(args.suppress_weighted))
    if args.suppress_weighted:
        logger.info('\tFading rate:      \t{}'.format(args.suppress_fading))
        logger.info('\tAmplifying rate:  \t{}'.format(args.suppress_increasing))

    logger.info('\tSuppress lazy:    \t{}'.format(args.suppress_lazy))
    logger.info('\tFitlering thresh: \t{}'.format(args.filtering_thresh))

    # if args.reg_explanations:
    """
    Language model is used in the explanation method for generating the neighboring instances 
    """
    train_lm_dataloder = processor.get_dataloader('train', args.train_batch_size)
    dev_lm_dataloader = processor.get_dataloader('dev', args.train_batch_size)
    explainer = SamplingAndOcclusionExplain(model, args, tokenizer, device=device, vocab=tokenizer.vocab,
                                            train_dataloader=train_lm_dataloder,
                                            dev_dataloader=dev_lm_dataloader,
                                            lm_dir=args.lm_dir,
                                            output_path=os.path.join(args.output_dir,
                                                                     args.output_filename),

                                            )
    # else:
    #     explainer = None

    if args.do_train:
        epoch = 0
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args, verbose=0)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        class_weight = torch.FloatTensor([args.negative_weight, 1]).to(device)

        model.train()
        losses = []
        reg_losses = []
        suppress_records = []
        for epoch_idx in range(int(args.num_train_epochs)):
            logger.info("***** Epoch {} *****".format(epoch_idx))
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss(class_weight)
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # regularize explanations
                # NOTE: backward performed inside this function to prevent OOM

                if args.reg_explanations:
                    if args.neutral_words_file != '':
                        reg_loss, reg_cnt = explainer.compute_explanation_loss(input_ids, input_mask, segment_ids, label_ids,
                                                                               do_backprop=True)
                    else:
                        reg_loss, reg_cnt = explainer.suppres_explanation_loss(input_ids, input_mask, segment_ids, label_ids,
                                                                               do_backprop=True)
                        tr_reg_loss += reg_loss  # float
                        tr_reg_cnt += reg_cnt
                else:
                    reg_loss = 0.
                losses.append(loss.item())
                reg_losses.append(loss.item() + reg_loss)

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                '''
                if global_step % 200 == 0:
                    val_result = validate(args, model, processor, tokenizer, output_mode, label_list, device,
                                          num_labels, task_name, tr_loss, global_step, epoch, explainer)
                    # validate current model & update the suppressing list
                    val_acc, val_f1 = val_result['acc'], val_result['f1']
                    if val_f1 > val_best_f1:
                        val_best_f1 = val_f1
                        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                            save_model(args, model, tokenizer, num_labels)
                    else:
                        # halve the learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        early_stop_countdown -= 1
                        logger.info("Reducing learning rate... Early stop countdown %d" % early_stop_countdown)
                    if early_stop_countdown < 0:
                        break
                    if global_step > args.max_iter > 0:
                        break
                '''
                if global_step % 200 == 0:
                    logger.info('***** Steps already made: {} *****'.format(global_step))
                    val_result = validate(args, model, processor, tokenizer, output_mode, label_list, device,
                                          num_labels, task_name, tr_loss, global_step, epoch, explainer)
                    val_acc, val_f1 = val_result['acc'], val_result['f1']
                    if val_f1 > val_best_f1:
                        val_best_f1 = val_f1
                        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                            save_model(args, model, tokenizer, num_labels)
                    else:
                        # halve the learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        early_stop_countdown -= 1
                        logger.info("Reducing learning rate... Early stop countdown %d" % early_stop_countdown)

                    update_suppress_list(args, model, processor, tokenizer, output_mode, label_list, device, explainer, train=1)
                    suppress_records.append(explainer.get_suppress_words())

                    if early_stop_countdown < 0:
                        break
                    if global_step > args.max_iter > 0:
                        break
            if early_stop_countdown < 0:
                break
            if global_step > args.max_iter > 0:
                logger.info('Maximum iteration criteria satisfied, Exit')
                break
            epoch += 1

        records = [losses, reg_losses, suppress_records]
        with open(args.stats_file, 'wb') as f:
            pickle.dump(records, f)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not args.explain:
            validate(args, model, processor, tokenizer, output_mode, label_list, device, num_labels,
                     task_name, tr_loss, global_step=0, epoch=-1, explainer=explainer)
        else:
            explain(args, model, processor, tokenizer, output_mode, label_list, device)


def update_suppress_list(args, model, processor, tokenizer, output_mode, label_list, device, explainer, train=0, verbose=0):
    model.train(False)
    if train:
        eval_examples = processor.get_train_examples(args.data_dir)
    elif args.test:
        eval_examples = processor.get_dev_examples(args.data_dir)
    else:
        eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args, verbose=0)

    logger.info("***** Updating suppress list *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    wrong_li = [[] for _ in range(4)]
    right_li = [[] for _ in range(4)]
    for i, input_batch in enumerate(eval_dataloader):
        idx_li = find_incorrect_pred(model, input_batch, device)
        idx_all = list(range(len(input_batch[0])))
        for j in range(len(wrong_li)):
            wrong_li[j] += [input_batch[j][idx] for idx in idx_li]
            right_li[j] += [input_batch[j][idx] for idx in idx_all if idx not in idx_li]

    if args.suppress_lazy:
        explainer.update_suppress_words_lazy([wrong_li[0], wrong_li[-1]], [right_li[0], right_li[-1]], verbose=verbose)
    else:
        explainer.update_suppress_words([wrong_li[0], wrong_li[-1]], [right_li[0], right_li[-1]], verbose=verbose)

    model.train(True)


def validate(args, model, processor, tokenizer, output_mode, label_list, device, num_labels,
             task_name, tr_loss, global_step, epoch, explainer=None):
    if not args.test:
        eval_examples = processor.get_dev_examples(args.data_dir)
    else:
        eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args, verbose=0)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.train(False)
    eval_loss, eval_loss_reg = 0, 0
    eval_reg_cnt = 0
    nb_eval_steps = 0
    preds = []

    # for detailed prediction results
    input_seqs = []

    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()

        if args.reg_explanations:
            with torch.no_grad():
                if args.neutral_words_file != '':
                    reg_loss, reg_cnt = explainer.compute_explanation_loss(input_ids, input_mask, segment_ids, label_ids,
                                                                           do_backprop=False)
                else:
                    reg_loss, reg_cnt = explainer.suppres_explanation_loss(input_ids, input_mask, segment_ids, label_ids,
                                                                           do_backprop=False)

            eval_loss_reg += reg_loss
            eval_reg_cnt += reg_cnt

        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        for b in range(input_ids.size(0)):
            i = 0
            while i < input_ids.size(1) and input_ids[b,i].item() != 0:
                i += 1
            token_list = tokenizer.convert_ids_to_tokens(input_ids[b,:i].cpu().numpy().tolist())
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
    with open(output_detail_file,'w') as writer:
        for i, seq in enumerate(input_seqs):
            pred = preds[i]
            gt = all_label_ids[i]
            writer.write('{}\t{}\t{}\n'.format(gt, pred, seq))

    model.train(True)
    return result


def explain(args, model, processor, tokenizer, output_mode, label_list, device):
    """
    Added into run_model.py to support explanations
    :param args: configs, or args
    :param model: The model to be explained
    :param processor: For explanations on Gab/WS etc. Dataset, take an instance of Processor as input.
                    See Processor for details about the processor
    :param tokenizer: The default BERT tokenizer
    :param output_mode: "classification" for Gab
    :param label_list: "[0,1]" for Gab
    :param device: An instance of torch.device
    :return:
    """
    assert args.eval_batch_size == 1
    processor.set_tokenizer(tokenizer)

    if args.algo == 'soc':
        try:
            train_lm_dataloder = processor.get_dataloader('train', args.train_batch_size)
            dev_lm_dataloader = processor.get_dataloader('dev', args.train_batch_size)
        except FileNotFoundError:
            train_lm_dataloder = None
            dev_lm_dataloader = None

        explainer = SamplingAndOcclusionExplain(model, args, tokenizer, device=device, vocab=tokenizer.vocab,
                                                train_dataloader=train_lm_dataloder,
                                                dev_dataloader=dev_lm_dataloader,
                                                lm_dir=args.lm_dir,
                                                output_path=os.path.join(args.output_dir, args.output_filename),
                                                )
    else:
        raise ValueError

    label_filter = None
    if args.only_positive and args.only_negative:
        label_filter = None
    elif args.only_positive: label_filter = 1
    elif args.only_negative: label_filter = 0

    if not args.test:
        eval_examples = processor.get_dev_examples(args.data_dir, label=label_filter)
    else:
        eval_examples = processor.get_test_examples(args.data_dir, label=label_filter)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.hiex_idxs:
        with open(args.hiex_idxs) as f:
            hiex_idxs = json.load(f)['idxs']
            print('Loaded line numbers for explanation')
    else:
        hiex_idxs = []

    model.train(False)
    for i, (input_ids, input_mask, segment_ids, label_ids) in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
        if i == args.stop: break
        if hiex_idxs and i not in hiex_idxs: continue
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        if not args.hiex:
            explainer.word_level_explanation_bert(input_ids, input_mask, segment_ids, label_ids)
        else:
            explainer.hierarchical_explanation_bert(input_ids, input_mask, segment_ids, label_ids)
    if hasattr(explainer, 'dump'):
        explainer.dump()

def save_model(args, model, tokenizer, num_labels):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)


if __name__ == "__main__":
    main()
