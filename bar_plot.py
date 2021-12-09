import numpy as np
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize


KEY_WORD = 'fidelity'
YLABEL = 'ratio'


class MyFidelity:
    def __init__(self):
        self.fidelity_list = []
        self.fidelity_min_list = []
        self.avg_fidelity = 0.
        self.avg_fidelity_min = 0.


def get_acc(_path):
    _f = open(_path, 'r')
    _state = 0
    _buff = MyFidelity()
    _blocks = []
    for _l in _f.readlines():
        if _l[0] == '=':
            # start of a block
            _state = 0
            _buff = MyFidelity()
            continue

        if _l[0] == '*':
            _state = 1
            continue

        if KEY_WORD in word_tokenize(_l):
            _segs = _l.split(',')
            _fidelity = float(_segs[0].split(':')[-1])
            _fidelity_min = float(_segs[1].split(':')[-1])
            if _state == 0:
                _buff.fidelity_list.append(_fidelity)
                _buff.fidelity_min_list.append(_fidelity_min)
            if _state == 1:
                _buff.avg_fidelity = _fidelity
                _buff.avg_fidelity_min = _fidelity_min
                _blocks.append(_buff)
    _f.close()
    return _blocks


def plot_2grp_2subgrp(bar_width, x_axis, v_li_li, x_tick_labels, labels=None, colors = ['royalblue', 'lightskyblue', 'indianred', 'lightcoral'], title=''):	# v_li_li: list of value lists
    fig, ax = plt.subplots()

    if labels is None:
        labels = ['' for _ in range(len(v_li_li))]
    """
    Pick your colors, and enable the two commented lines if 2+2 mode is wanted
    """
    ax.bar(x_axis, color=colors[0], height=v_li_li[0], width=bar_width, label=labels[0])
    ax.bar(x_axis+bar_width*0.75, color=colors[1], height=v_li_li[1], width=bar_width/2, label=labels[1])
    ax.bar(x_axis + 2 * bar_width, color=colors[2], height=v_li_li[2], width=bar_width, label=labels[2])
    ax.bar(x_axis + bar_width*2.75, color=colors[3], height=v_li_li[3], width=bar_width/2, label=labels[3],
           alpha=0.6)
    ax.set_xticks(x_axis+bar_width*1.25)
    ax.set_xticklabels(x_tick_labels)
    ax.set_title(title)
    # plt.axhline(y=0.5, xmin=0., xmax=4., linestyle='--', color='black', alpha=0.2)
    plt.ylabel('fidelity')
    plt.legend(loc='best')
    plt.show()


def plot_2grp(bar_width, x_axis, v_li_li, x_tick_labels, labels=None, colors = ['royalblue', 'indianred'], title=''):	# v_li_li: list of value lists
    fig, ax = plt.subplots()

    if labels is None:
        labels = ['' for _ in range(len(v_li_li))]
    """
    Pick your colors, and enable the two commented lines if 2+2 mode is wanted
    """
    ax.bar(x_axis, color=colors[0], height=v_li_li[0], width=bar_width, label=labels[0])
    ax.bar(x_axis + 1.3 * bar_width, color=colors[1], height=v_li_li[1], width=bar_width, label=labels[1])
    ax.set_xticks(x_axis+bar_width*0.65)
    ax.set_xticklabels(x_tick_labels)
    ax.set_title(title)
    plt.axhline(y=0.5, xmin=0., xmax=4., linestyle='--', color='black', alpha=0.2)
    plt.ylabel(YLABEL)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    idf_list = ['racist', 'islam', 'jew', 'fucking', 'black', 'cu']
    value_list1 = [0.7333, 0.5556, 0.44, 0.5714, 0.5, 0.5714]
    value_list0 = [0., 0.2, 0., 0.1333, 0.2308, 0.1818]

    x_axis = np.array(range(len(idf_list)))
    bar_width = 0.25

    plot_2grp(bar_width, x_axis, [value_list0, value_list1], labels=['FNR', 'FPR'], x_tick_labels=idf_list)

    

