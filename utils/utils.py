from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from loader import convert_examples_to_features
import re
import os
import torch
import json
import yaml

CONFIG_NAME = "config.json"     # TODO: do multiple config to separate model from framework
WEIGHTS_NAME = "pytorch_model.bin"
PHASE_NAMES = ['normal', 'correcting', 'stabilizing']
MAX_LINE_WIDTH = 150


def load_config(pth):
    with open(pth, 'r') as f:
        return yaml.full_load(f)


def load_text_as_feature(args, processor, tokenizer, dataset, output_mode='classification'):
    valid_choices = ['train', 'test', 'eval']
    assert dataset in valid_choices, 'Invalid dataset is given: [{}], valid choices {}'.format(dataset, valid_choices)
    if dataset == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif dataset == 'eval':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(examples, processor.get_labels(),
                                            args.max_seq_length, tokenizer, output_mode, args, verbose=0)
    return features, examples


def load_word_from_file(pth):
    with open(pth) as f:
        words = []
        for line in f.readlines():
            segs = line.strip().split('\t')
            word = segs[0]
            words.append(word)
        assert words
    return words


def record_attr_change(soc, input_pack, attr_obj, steps, key='test'):
    target = attr_obj.idx_test_dict if key == 'test' else attr_obj.idx_train_dict
    all_ids, all_mask, all_segments = input_pack
    scores = []
    for instance_id in target.keys():
        for pos in target[instance_id]:
            x_region = (pos, pos)
            scores.append(get_explanation(soc, x_region, all_ids[instance_id],
                                          all_mask[instance_id], all_segments[instance_id]))
    attr_obj.record_attr_change(steps, scores)


def get_explanation(soc, x_region, input_ids, input_mask, segment_ids):
    score = soc.algo.do_attribution(input_ids, input_mask, segment_ids, x_region)
    return score


def find_positions(all_inputs, attr_obj):
    positions = (all_inputs == attr_obj.id).nonzero(as_tuple=False)
    for pos in positions:
        attr_obj.update_test_dict(pos[0].item(), pos[1].item())


def compute_metrics(preds, labels, pred_probs):
    assert len(preds) == len(labels), \
        'Unmatched length between predictions [{}] and ground truth [{}]'.format(len(preds), len(labels))
    return acc_and_f1(preds, labels, pred_probs)


def acc_and_f1(preds, labels, pred_probs):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(y_true=labels, y_pred=preds)
    p, r = precision_score(y_true=labels, y_pred=preds), recall_score(y_true=labels, y_pred=preds)
    try:
        roc = roc_auc_score(y_true=labels, y_score=pred_probs[:, 1])
    except ValueError:
        roc = 0.
    return {
        "acc": acc, "f1": f1, "precision": p, "recall": r, "auc_roc": roc
    }


class AttrRecord:
    def __init__(self, w, w_id):
        self.w = w
        self.id = w_id

        self.checked_test = False
        self.checked_train = False

        # Key of the dict is the index of instances containing the target word
        # The value is a list of positions where the word locates
        self.idx_test_dict = dict()
        self.idx_train_dict = dict()
        self.attr_changes = OrderedDict()   # values are list of attributions for instances
        self.fpr_changes = OrderedDict()    # values are tuples of #FP #TP #FPR

    def record_attr_change(self, steps, attr):
        self.attr_changes[steps] = attr

    def record_fpr_change(self, steps, attr):
        self.fpr_changes[steps] = attr

    def update_test_dict(self, idx, pos):
        self.checked_test = True
        if idx in self.idx_test_dict:
            self.idx_test_dict[idx].append(pos)
        else:
            self.idx_test_dict[idx] = [pos]

    def update_train_dict(self, idx, pos):
        self.checked_train = True
        if self.idx_train_dict:
            self.idx_train_dict[idx].append(pos)
        else:
            self.idx_train_dict[idx] = [pos]

    def disable_test_dict(self):
        self.idx_test_dict = None

    def disable_train_dict(self):
        self.idx_train_dict = None

    def get_test_idxs(self):
        return self.idx_test_li

    def get_train_idxs(self):
        return self.idx_train_li

    def get_change(self):
        epoch_li = [epoch for epoch in self.attr_changes.keys()]
        attr_li = [self.attr_changes[epoch] for epoch in epoch_li]
        return epoch_li, attr_li


def save_model(args, model, tokenizer, phase=None, postfix=None):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    if phase is None:
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        target_dir = args.output_dir
    else:
        target_dir = args.output_dir + '_' + PHASE_NAMES[phase] + postfix
        output_model_file = os.path.join(target_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(target_dir, CONFIG_NAME)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(target_dir)
    if args.do_train:
        f = open(os.path.join(target_dir, 'args.json'), 'w')
        json.dump(args.__dict__, f, indent=4)
        f.close()


def seconds2hms(s):
    h = s//3600
    m = (s % 3600) // 60
    s = s % 60
    return h, m, s


def heading(msg):
    remains = MAX_LINE_WIDTH - len(msg) - 2
    return '|' + ' '*(remains // 2) + msg + ' '*(remains // 2 + remains % 2) + '|'


class DescStr:
    def __init__(self):
        self._desc = ''

    def write(self, instr):
        self._desc += re.sub('\n|\x1b.*|\r', '', instr)

    def read(self):
        ret = self._desc
        self._desc = ''
        return ret

    def flush(self):
        pass
