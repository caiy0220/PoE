from collections import OrderedDict


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


def find_positions(all_inputs, attr_obj, key='test'):
    positions = (all_inputs == attr_obj.id).nonzero(as_tuple=False)
    for pos in positions:
        if key == 'test':
            attr_obj.update_test_dict(pos[0].item(), pos[1].item())


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
