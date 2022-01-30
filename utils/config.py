"""
This file contain configs for training models and running
"""
import logging
import argparse

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        """
        Deprecated: Every arguments here are by default overrode by commandline arguments. (See run_model.py)
        """

        self.raw_data_path = './data/majority_gab_dataset_25k.json'  # default using MAJORITY
        self.data_dir = './data/majority_gab_dataset_25k/'

        # create a classifier head for each item in the list with disjunction of each label
        # for example, [('hd','cv')] means the ground truth label for classification is 1 when hd OR cv is 1.
        # multi-head classifier not implemented. So len(self.label_groups) == 1 should hold.
        self.label_groups = [('hd', 'cv')]
        self.do_lower_case = True
        self.bert_model = 'bert-base-uncased'

        # for hierarchical explanation algorithms, where to store the language model
        # first time running may require to train a language model
        self.lm_model_dir = 'runs/lm_model_uncased.pkl'

        # configs for lm in hierarchical explanation algorithms
        self.lm_d_hidden = 300
        self.lm_d_embed = 300
        self.batch_size = 1

        # context region to be specified for running SOC. Smaller to be faster and larger to be better.
        self.nb_range = 20
        # the number of samples to be drawn for SOC. Smaller to be faster and larger to be better.
        self.sample_n = 20

        # keep self.max_seq_length identical for that training bert. Both are 128 by default
        self.max_seq_length = 128

        # whether pad the words outside the context region of a given phrase to be explained
        # when turned TRUE, SOC yields completely global explanations, and achieve better correlation
        # with word importance captured by linear TF-IDF model.
        # when turned FALSE, SOC explanations are global to its contexts, but local (specific) to the
        # words outside the context region. It has better mathematical interpretation, but achieve lower
        # correlation with word importance captured by linear TF-IDF model
        # NOTE: only configure this with command line
        self.mask_outside_nb = False

        # whether pad the context of the phrase instead of sampling. Turning
        # NOTE: only configure this with command line
        self.use_padding_variant = False

        # neutral words
        self.neutral_words_file = 'data/identity.csv'
        self.reg_mse = True
        self.reg_strength = 0.1

        # whether keep other neutral words during regularization
        self.keep_other_nw = True

        # whether do neutral word removal
        self.remove_nw = False

    def update(self, other):
        combine_args(self, other)


configs = Config()


def combine_args(args, other):
    # combine and update configs, skip if args.<k> is None
    for k, v in other.__dict__.items():
        if getattr(other, k) is not None:
            if hasattr(args, k) and getattr(args, k) != v:
                logger.info('Overriding {} from {} to {})'.format(k, getattr(args, k), v))
            setattr(args, k, v)


def get_new_parser():
    return argparse.ArgumentParser()


def add_general_configs(parser, **kwargs):
    general_params = parser.add_argument_group('general config')
    # Required parameters
    general_params.add_argument("--data_dir", default=None, type=str, required=True,
                                help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    general_params.add_argument("--bert_model", default=None, type=str, required=True,
                                help="Bert pre-trained model selected in the list: bert-base-uncased, " 
                                     "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, " 
                                     "bert-base-multilingual-cased, bert-base-chinese.")
    general_params.add_argument("--task_name", default=None, type=str, required=True,
                                help="The name of the task to train.")
    general_params.add_argument("--output_dir", default=None, type=str, required=True,
                                help="The output directory where the model predictions and checkpoints will be written.")

    general_params.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    general_params.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # if true, use test data instead of val data
    general_params.add_argument("--test", action='store_true')
    general_params.add_argument("--negative_weight", default=1., type=float)

    # whether run explanation algorithms
    general_params.add_argument("--explain", action='store_true', help='if true, explain test set predictions')
    general_params.add_argument("--debug", action='store_true')

    # the output filename without postfix
    general_params.add_argument("--output_filename", default='temp.tmp')
    # the directory where the lm is stored
    general_params.add_argument("--lm_dir", default='runs/lm')
    general_params.add_argument("--stats_file", default='log.pkl')
    general_params.add_argument("--cache_dir", default="", type=str,
                                help="Where do you want to store the pre-trained models downloaded from s3")

    # Choose working mode
    general_params.add_argument("--do_train", action='store_true', help="Whether to run training.")
    general_params.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    # ---------------------- Seldom used ----------------------
    # if true, generate hierarchical explanations instead of word level outputs.
    # Only useful when the --explain flag is also added.
    general_params.add_argument("--hiex", action='store_true')
    general_params.add_argument("--hiex_tree_height", default=5, type=int)
    # whether add the sentence itself to the sample set in SOC
    general_params.add_argument("--hiex_add_itself", action='store_true')
    # if configured, only generate explanations for instances with given line numbers
    general_params.add_argument("--hiex_idxs", default=None)
    # if true, use absolute values of explanations for hierarchical clustering
    general_params.add_argument("--hiex_abs", action='store_true')

    # if either of the two is true, only generate explanations for positive / negative instances
    general_params.add_argument("--only_positive", action='store_true')
    general_params.add_argument("--only_negative", action='store_true')
    # stop after generating x explanation
    general_params.add_argument("--stop", default=100000000, type=int)

    general_params.add_argument("--reg_mse", action='store_true')

    general_params.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    general_params.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    return parser

def add_explanation_configs(parser, **kwargs):
    explanation_params = parser.add_argument_group('Explanation configs')

    # weight of regularization term
    explanation_params.add_argument("--reg_strength", type=float)

    # which algorithm to run
    explanation_params.add_argument("--algo", choices=['soc'])

    explanation_params.add_argument("--neutral_words_file", default='')
    explanation_params.add_argument("--neg_suppress_file", default='')
    explanation_params.add_argument("--pos_suppress_file", default='')

    # see utils/config.py
    explanation_params.add_argument("--use_padding_variant", action='store_true')
    explanation_params.add_argument("--mask_outside_nb", action='store_true')
    explanation_params.add_argument("--nb_range", type=int, default=5)
    explanation_params.add_argument("--sample_n", type=int, default=5)

    # configuring explanation regularization
    explanation_params.add_argument("--reg_explanations", action='store_true')
    explanation_params.add_argument("--reg_balanced", action='store_true')

    # Defining how over-represented features should be selected
    # If suppress lazy is set as true, only features constantly being biased will be picked,
    # otherwise the suppressing list will be updated regularly after each validation
    explanation_params.add_argument("--suppress_lazy", action='store_true')
    explanation_params.add_argument("--window_size", type=int, default=5)
    explanation_params.add_argument("--ratio_in_window", type=float, default=0.6)   # definition of "constantly"
    # Threshold of defining over-represented features/words
    explanation_params.add_argument("--filtering_thresh", type=float, default=0.7)

    # configuring whether the weights of words should change,
    # and if yes, how would they change
    explanation_params.add_argument("--suppress_weighted", action='store_true')
    explanation_params.add_argument("--suppress_fading", type=float, default=0.7)
    explanation_params.add_argument("--suppress_increasing", type=float, default=1.2)
    explanation_params.add_argument("--suppress_lower_thresh", type=float, default=0.5)
    explanation_params.add_argument("--suppress_higher_thresh", type=float, default=2.)

    # whether discard other neutral words during regularization. default: False
    # explanation_params.add_argument("--discard_other_nw", action='store_false', dest='keep_other_nw')
    return parser


def add_training_configs(parser, **kwargs):
    training_params = parser.add_argument_group('Training configs')

    # whether remove neutral words when loading datasets
    training_params.add_argument("--remove_nw", action='store_true')

    training_params.add_argument("--learning_rate", default=5e-5, type=float,
                                 help="The initial learning rate for Adam.")

    # early stopping with decreasing learning rate. 0: direct exit when validation F1 decreases
    training_params.add_argument("--early_stop", default=5, type=int)
    # If max iter is set to be positive, early_stop will be over writen
    training_params.add_argument("--max_iter", default=-1, type=int)
    # epochs to go
    training_params.add_argument("--num_train_epochs", default=3.0, type=float,
                                 help="Total number of training epochs to perform.")

    training_params.add_argument("--train_batch_size", default=32, type=int,
                                 help="Total batch size for training.")
    training_params.add_argument("--eval_batch_size", default=32, type=int,
                                 help="Total batch size for eval.")
    training_params.add_argument("--warmup_proportion", default=0.1, type=float,
                                 help="Proportion of training to perform linear learning rate warmup for. " 
                                      "E.g., 0.1 = 10%% of training.")
    training_params.add_argument("--local_rank", type=int, default=-1,
                                 help="local_rank for distributed training on gpus")
    training_params.add_argument('--gradient_accumulation_steps', type=int, default=1,
                                 help="Number of updates steps to accumulate before performing a backward/update pass.")
    training_params.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    training_params.add_argument('--loss_scale', type=float, default=0,
                                 help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n" 
                                      "0 (default value): dynamic loss scaling.\n" 
                                      "Positive power of 2: static loss scaling value.\n")

    # preprocessing settings
    training_params.add_argument("--max_seq_length", default=128, type=int,
                                 help="The maximum total input sequence length after WordPiece tokenization. \n" 
                                      "Sequences longer than this will be truncated, and sequences shorter \n" 
                                      "than this will be padded.")
    training_params.add_argument("--do_lower_case", action='store_true',
                                 help="Set this flag if you are using an uncased model.")
    return parser
