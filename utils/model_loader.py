from bert.tokenization import BertTokenizer
from bert.optimization import BertAdam
from loader import GabProcessor, WSProcessor, NytProcessor


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


def get_optimizer(args, model):
    if not args.do_train:
        return None
    num_phases = 3 if args.mode == 'mid' else 1
    num_train_optimization_steps = args.max_iter * num_phases

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                    warmup=args.warmup_proportion, t_total=num_train_optimization_steps)
