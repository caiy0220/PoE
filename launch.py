from __future__ import absolute_import, division, print_function

import logging
import os
import random
import utils.utils as my_utils
import utils.model_loader as model_loader
import argparse

import numpy as np
import torch

from bert.modeling import BertForSequenceClassification

from hiex import SamplingAndOcclusionExplain
from poe import MiD

# TODO: terminal control of parser

VERSION = 'Current version of main: 1.000.001'

try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                   Path.home() / '.pytorch_pretrained_bert'))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                              os.path.join(os.path.expanduser("~"), '.pytorch_pretrained_bert'))


logger = logging.getLogger(__name__)


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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    """ 
    ------------------------------------------------------------
    |                Prepare training setting                  |
    ------------------------------------------------------------
    """
    device, n_gpu = get_hardware_setting(args)
    # logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #     device, n_gpu, bool(args.local_rank != -1), args.fp16))
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    assert args.gradient_accumulation_steps >= 1, 'Invalid gradient_accumulation_steps parameter, should be >= 1'
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    assert args.do_train ^ args.do_eval, 'Activate either do_train or do_eval at a time'

    set_random_seed(args)

    """ 
    ------------------------------------------------------------
    |             Load model and prepare dataset               |
    ------------------------------------------------------------
    """
    tokenizer, processor = model_loader.get_processors(args)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)
    model = model.to(device)
    optimizer = model_loader.get_optimizer(args, model)

    """ 
    ------------------------------------------------------------
    |               Load SOC which deploys LM                  |
    ------------------------------------------------------------
    """
    explainer = SamplingAndOcclusionExplain(model, args, tokenizer, processor, device=device, lm_dir=args.lm_dir,
                                            output_path=os.path.join(args.output_dir, args.output_filename))

    train_features, train_examples = my_utils.load_text_as_feature(args, processor, tokenizer, 'train')
    eval_features, eval_examples = my_utils.load_text_as_feature(args, processor, tokenizer, 'eval')
    # explainer.get_global_words_count(np.array([f.input_ids for f in train_features], dtype=np.int64))    # Preparation for FPP

    """ 
    ------------------------------------------------------------
    |                   Start Training                         |
    ------------------------------------------------------------
    """
    mid = MiD(args, device, logger=logger, tokenizer=tokenizer, processor=processor)
    # mid.load_tools(tokenizer, processor, logger)
    mid.load_model(model, optimizer)
    mid.load_explainer(explainer)
    mid.load_data(train_features, eval_features)

    logger.info('\n')
    logger.info('='*my_utils.MAX_LINE_WIDTH)

    mid.train()


if __name__ == '__main__':
    pref = 'utils/'
    files = ['cfg.yaml', 'lm.yaml', 'soc.yaml']
    t_args = argparse.Namespace()
    for file in files:
        t_args.__dict__.update(my_utils.load_config(pref + file))

    parser = argparse.ArgumentParser()
    _args = parser.parse_args(namespace=t_args)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if _args.local_rank in [-1, 0] else logging.WARN)
    logger.info(my_utils.heading(VERSION))

    logger.info('='*my_utils.MAX_LINE_WIDTH)
    logger.info('{}'.format(_args))
    logger.info('='*my_utils.MAX_LINE_WIDTH)

    main(_args)
