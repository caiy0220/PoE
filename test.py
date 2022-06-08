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
import argparse
import utils.utils as my_utils
import utils.model_loader as model_loader

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from poe import unpack_features
from torch.nn import functional as F
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from bert.modeling import BertForSequenceClassification
from hiex import SamplingAndOcclusionExplain


logger = logging.getLogger(__name__)
VERSION = 'Current version of test: 1.000.000'


def evaluate(args, processor, tokenizer, device, dl, phase=None, postfix=None):
    output_mode = 'classification'

    # Prepare model
    model_dir = args.output_dir if phase is None else args.output_dir + '_' + my_utils.PHASE_NAMES[phase] + postfix
    logger.info('\n\n')
    logger.info(my_utils.heading('Loading model from [{}]'.format(model_dir)))
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=len(processor.get_labels()))
    model.to(device)

    """
    Language model is used in the explanation method for generating the neighboring instances 
    """
    explainer = SamplingAndOcclusionExplain(model, args, tokenizer, processor, device=device, lm_dir=args.lm_dir,
                                            output_path=os.path.join(args.output_dir, args.output_filename))

    if not args.explain:
        validate(args, model, processor, tokenizer, dl, device, explainer=explainer)
    # else:
    #     explain(args, model, processor, tokenizer, output_mode, label_list, device)


def validate(args, model, processor, tokenizer, dl, device, explainer=None):
    num_labels = len(processor.get_labels())

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dl.dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.train(False)
    eval_loss, eval_loss_reg = 0, 0
    eval_reg_cnt, nb_eval_steps = 0, 0
    preds, ys = [], []

    # for detailed prediction results
    input_seqs = []
    loss_fct = CrossEntropyLoss()

    for step, batch in enumerate(tqdm(dl, desc='#Eval')):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(batch[0], batch[2], batch[1], labels=None)  # be careful with the order

        # create eval loss and other metric required by the task
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), batch[-1].view(-1))

        eval_loss += tmp_eval_loss.mean().item()

        with torch.no_grad():
            if args.mode == 'mid':
                reg_loss, reg_cnt = explainer.suppress_explanation_loss(*batch, do_backprop=False)
            elif args.mode == 'soc':
                reg_loss, reg_cnt = explainer.compute_explanation_loss(*batch, do_backprop=False)
            else:
                reg_loss, reg_cnt = 0, 0

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
            token_list = tokenizer.convert_ids_to_tokens(batch[0][b, :i].cpu().numpy().tolist())
            input_seqs.append(' '.join(token_list))

    logger.info('\n')

    eval_loss = eval_loss / nb_eval_steps
    eval_loss_reg = eval_loss_reg / (eval_reg_cnt + 1e-10)

    preds = preds[0]
    ys = ys[0]
    pred_labels = np.argmax(preds, axis=1)
    pred_prob = F.softmax(torch.from_numpy(preds).float(), -1).numpy()

    result = my_utils.compute_metrics(pred_labels, ys, pred_prob)
    loss = None
    global_step = -1

    result['eval_loss'] = eval_loss
    result['eval_loss_reg'] = eval_loss_reg
    result['loss'] = loss

    split = 'test'

    output_eval_file = os.path.join(args.output_dir, "eval_results_%d_%s_%s.txt"
                                    % (global_step, split, args.task_name))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
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
            gt = ys[i]
            writer.write('{}\t{}\t{}\n'.format(gt, pred, seq))

    model.train(True)
    return result


# def explain(args, model, processor, tokenizer, output_mode, label_list, device):
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
    '''
    assert args.eval_batch_size == 1
    processor.set_tokenizer(tokenizer)

    if args.algo == 'soc':
        explainer = SamplingAndOcclusionExplain(model, args, tokenizer, processor, device=device, lm_dir=args.lm_dir,
                                                output_path=os.path.join(args.output_dir, args.output_filename))
    else:
        raise ValueError

    label_filter = None
    if args.only_positive and args.only_negative:
        label_filter = None
    elif args.only_positive:
        label_filter = 1
    elif args.only_negative:
        label_filter = 0

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

    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

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
    '''


def get_hardware_setting(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    return device, n_gpu


def set_random_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def main(args):
    device, n_gpu = get_hardware_setting(args)
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    assert args.gradient_accumulation_steps >= 1, 'Invalid gradient_accumulation_steps parameter, should be >= 1'
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    set_random_seed(args)

    tokenizer, processor = model_loader.get_processors(args)

    if args.mode == 'mid':
        l, r = 0, 3
    elif args.mode == 'soc':
        l, r = 1, 2
    else:
        l, r = 0, 1

    eval_features, eval_examples = my_utils.load_text_as_feature(args, processor, tokenizer, 'test')
    eval_ds = unpack_features(eval_features, args)
    data = TensorDataset(*eval_ds)
    sampler = SequentialSampler(data)
    eval_dl = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

    for phase in range(l, r):
        evaluate(args, processor, tokenizer, device, eval_dl, phase, '_best')
        evaluate(args, processor, tokenizer, device, eval_dl, phase, '_final')
        logger.info('='*my_utils.MAX_LINE_WIDTH)


if __name__ == "__main__":
    pref = 'utils/'
    files = ['cfg.yaml', 'lm.yaml', 'soc.yaml']
    config = dict()
    for file in files:
        config.update(my_utils.load_config(pref + file))

    parser = argparse.ArgumentParser()
    ns = argparse.Namespace()
    ns.__dict__.update(config)
    for k in ns.__dict__:
        parser.add_argument('--'+k, type=type(getattr(ns, k)))

    _args = parser.parse_args(namespace=ns)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if _args.local_rank in [-1, 0] else logging.WARN)

    logger.info(VERSION)
    logger.info('='*my_utils.MAX_LINE_WIDTH)
    logger.info('{}'.format(_args))
    args_diff = my_utils.get_args_diff(config, _args.__dict__)
    if args_diff:
        logger.info('-' * my_utils.MAX_LINE_WIDTH)
        logger.info('{}'.format(args_diff))
    logger.info('='*my_utils.MAX_LINE_WIDTH)

    main(_args)
