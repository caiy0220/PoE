from __future__ import absolute_import, division, print_function

import logging
import os
import random
import utils.utils as my_utils
import yaml
import argparse
# import pickle

import numpy as np
import torch
# from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
#                               TensorDataset)
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm

# from sklearn.metrics import matthews_corrcoef, f1_score
# from sklearn.metrics import precision_score, recall_score, roc_auc_score
# from bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME, PHASE_NAMES
from bert.modeling import BertForSequenceClassification
from bert.tokenization import BertTokenizer
from bert.optimization import BertAdam, WarmupLinearSchedule

from loader import GabProcessor, WSProcessor, NytProcessor, convert_examples_to_features
# import utils.config as config

from hiex import SamplingAndOcclusionExplain
from poe import MiD

logger = logging.getLogger(__name__)


def load_config(pth):
    with open(pth, 'r') as f:
        return yaml.full_load(f)


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


def get_processors(args):
    processors = {
        'gab': GabProcessor,
        'ws': WSProcessor,
        'nyt': NytProcessor
    }
    task_name = args.task_name.lower()
    assert task_name in processors, 'Task not found [{}]'.format(task_name)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    return tokenizer, processors[task_name](args, tokenizer=tokenizer)


def main(args):
    device, n_gpu = get_hardware_setting(args)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    assert args.gradient_accumulation_steps >= 1, 'Invalid gradient_accumulation_steps parameter, should be >= 1'
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    assert args.do_train ^ args.do_eval, 'Activate either do_train or do_eval at a time'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_mode = 'classification'
    """ 
    ------------------------------------------------------------
    |             Load model and prepare dataset               |
    ------------------------------------------------------------
    """
    tokenizer, processor = get_processors(args)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    model = BertForSequenceClassification.from_pretrained('notebook/corrected',
                                                          cache_dir=args.cache_dir,
                                                          num_labels=num_labels)
    model = model.to('cuda')

    # TODO: wrap the loading process with functions
    train_examples = processor.get_train_examples(args.data_dir)
    eval_examples = processor.get_dev_examples(args.data_dir)
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, output_mode, args, verbose=0)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, args, verbose=0)

    num_phases = 3 if args.mode == 'mid' else 1
    num_train_optimization_steps = args.max_iter * num_phases

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.do_train:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

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
                                                                     args.output_filename))


if __name__ == '__main__':
    pref = 'utils/'
    files = ['cfg.yaml', 'lm.yaml', 'soc.yaml']
    t_args = argparse.Namespace()
    for file in files:
        t_args.__dict__.update(load_config(pref + file))

    parser = argparse.ArgumentParser()
    _args = parser.parse_args(namespace=t_args)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if _args.local_rank in [-1, 0] else logging.WARN)
    logger.info('Current version of main: 0.000.213')

    main(_args)


# mid.model.train(False)
# eval_loss, eval_loss_reg = 0, 0
# eval_reg_cnt, nb_eval_steps = 0, 0
# preds = []
# ys = []
# # for detailed prediction results
# input_seqs = []
#
# # for input_ids, input_mask, segment_ids, label_ids in eval_dl:
# for step, batch in enumerate(tqdm(dl, desc='Validate')):
#     batch = tuple(t.to(mid.device) for t in batch)
#
#     with torch.no_grad():
#         logits = mid.model(batch[0], batch[2], batch[1], labels=None)  # be careful with the order
#     # loss_fct = CrossEntropyLoss()
#     # tmp_eval_loss = mid.loss_fct(logits.view(-1, mid.num_labels), batch[-1].view(-1))
#     tmp_eval_loss = loss_fct(logits.view(-1, mid.num_labels), batch[-1].view(-1))
#     eval_loss += tmp_eval_loss.mean().item()
#
#     if mid.args.reg_explanations:
#         with torch.no_grad():
#             if mid._mode == 'mid':
#                 reg_loss, reg_cnt = mid.explainer.suppress_explanation_loss(*batch, do_backprop=False)
#             else:
#                 reg_loss, reg_cnt = mid.explainer.compute_explanation_loss(*batch, do_backprop=False)
#         eval_loss_reg += reg_loss
#         eval_reg_cnt += reg_cnt
#     nb_eval_steps += 1
#     if len(preds) == 0:
#         preds.append(logits.detach().cpu().numpy())
#         ys.append(batch[-1].detach().cpu().numpy())
#     else:
#         preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
#         ys[0] = np.append(ys[0], batch[-1].detach().cpu().numpy(), axis=0)
#
#     for b in range(batch[0].size(0)):
#         i = 0
#         while i < batch[0].size(1) and batch[0][b, i].item() != 0:
#             i += 1
#         token_list = mid.tokenizer.convert_ids_to_tokens(batch[0][b, :i].cpu().numpy().tolist())
#         input_seqs.append(' '.join(token_list))
#
# eval_loss = eval_loss / nb_eval_steps
# eval_loss_reg = eval_loss_reg / (eval_reg_cnt + 1e-10)
# preds = preds[0]
# pred_labels = np.argmax(preds, axis=1)  # FIXME: Currently, only support classification
# pred_prob = torch.nn.functional.softmax(torch.from_numpy(preds).float(), -1).numpy()
# # result = my_utils.compute_metrics(pred_labels, all_label_ids.numpy(), pred_prob)
# # TODO: 0. test this
# result = my_utils.compute_metrics(pred_labels, ys[0], pred_prob)
# loss = tr_loss / (mid.global_step + 1e-10) if mid.args.do_train else None
#
# result['eval_loss'] = eval_loss
# result['eval_loss_reg'] = eval_loss_reg
# result['loss'] = loss
#
# split = 'dev'

