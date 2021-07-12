import torch
import random
import numpy as np
import argparse
from tqdm import tqdm, trange
import os
import datetime
import json
from model import MCWS
from transformers import BertConfig, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from utils import *


def train(args):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tag2index = {"S": 0, "B": 1, "I": 2, "E": 3}
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    train_data = read_file(args.train_data_path)
    train_data=train_data[:args.train_num]
    train_loader = data_generator(train_data, args.train_batch_size, tokenizer)
    eval_data = read_file(args.eval_data_path)
    eval_loader = data_generator(eval_data, args.eval_batch_size, tokenizer)
    word2id = get_word2id(args.train_data_path)
    first_device = torch.device(
        "cuda:{}".format(args.first_device_id) if torch.cuda.is_available() else "cpu")
    second_device = torch.device(
        "cuda:{}".format(args.second_device_id) if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_name is None:
        raise Warning('model name is not specified, the model will NOT be saved!')
    output_model_dir = os.path.join('./models', args.model_name + '_' + now_time)
    if not os.path.exists(output_model_dir):
        os.mkdir(output_model_dir)


    text_config = BertConfig.from_pretrained(args.bert_model)
    audio_config = BertConfig.from_json_file(args.audio_encoder_config_path)


    model = MCWS.from_pretrained(args.bert_model, config=text_config, audio_config=audio_config, args=args,
                                 first_device=first_device, second_device=second_device)

    epoch_step = len(train_data) // args.train_batch_size if len(train_data) % args.train_batch_size == 0 else len(train_data) // args.train_batch_size+1
    num_train_optimization_steps = epoch_step * args.num_train_epochs
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warm_up=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    best_epoch = -1
    best_p = -1
    best_r = -1
    best_f = -1
    best_oov = -1
    history = {'epoch': [], 'p': [], 'r': [], 'f': [], 'oov': []}

    for epoch in trange(args.num_train_epochs, desc="Epoch"):

        model.train()
        for input_ids, attention_mask, token_type_ids, audio_feature, audio_mask, labels in \
                tqdm(train_loader, desc="Training"):
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(first_device), attention_mask.to(
                first_device), token_type_ids.to(first_device), labels.to(first_device)

            audio_feature, audio_mask = audio_feature.to(first_device), audio_mask.to(first_device)

            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            audio_feature=audio_feature, audio_mask=audio_mask, labels=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, audio_feature, audio_mask, labels in tqdm(eval_loader,
                                                                                                    desc="Evaluating"):
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(first_device), attention_mask.to(
                    first_device), token_type_ids.to(first_device), labels.to(first_device)
                audio_feature, audio_mask = audio_feature.to(first_device), audio_mask.to(first_device)
                _, logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                  audio_feature=audio_feature, audio_mask=audio_mask, labels=labels)
                lengths = torch.sum(attention_mask, 1)
                for i, length in enumerate(lengths):
                    y_true.append(labels[i][0:length - 2])
                    y_pred.append(logits[i][0:length - 2])

            p, r, f = compute_scores(y_pred, y_true, tag2index)
            oov = cws_evaluate_OOV(y_pred, y_true, eval_data, word2id, tag2index)

        history['epoch'].append(epoch)
        history['p'].append(p)
        history['r'].append(r)
        history['f'].append(f)
        history['oov'].append(oov)


        print("\nEpoch: %d, P: %f, R: %f, F: %f, OOV: %f"%(epoch + 1, p, r, f, oov))

        if f > best_f:
            best_epoch = epoch + 1
            best_p = p
            best_r = r
            best_f = f
            best_oov = oov

            if args.model_name:
                with open(os.path.join(output_model_dir, 'CWS_result.txt'), "w") as writer:
                    for i in range(len(y_pred)):
                        seg_true_str, seg_pred_str = eval_sentence(y_pred[i], y_true[i], eval_data[i][0], tag2index)

                        writer.write('True: %s\n' % seg_true_str)
                        writer.write('Pred: %s\n\n' % seg_pred_str)

                best_eval_model_path = os.path.join(output_model_dir, 'model.pt')

                torch.save(model.state_dict(), best_eval_model_path)


    if os.path.exists(output_model_dir):
        with open(os.path.join(output_model_dir, 'history.json'), 'w', encoding='utf8') as f:
            json.dump(history, f)
            f.write('\n')


def test(args):
    tag2index = {"S": 0, "B": 1, "I": 2, "E": 3}
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    test_data = read_file(args.eval_data_path)
    test_loader = data_generator(test_data, args.eval_batch_size,tokenizer)
    first_device = torch.device(
        "cuda:{}".format(args.first_device_id) if torch.cuda.is_available() else "cpu")
    second_device = torch.device(
        "cuda:{}".format(args.second_device_id) if torch.cuda.is_available() else "cpu")
    text_config = BertConfig.from_pretrained(args.bert_model)
    audio_config = BertConfig.from_json_file(args.audio_encoder_config_path)

    model = MCWS(args.bert_model, config=text_config, audio_config=audio_config, args=args,
                                 first_device=first_device, second_device=second_device)
    model_checkpoint = torch.load(args.eval_model)
    model.load_state_dict(model_checkpoint)


    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, audio_feature, audio_mask, labels in tqdm(test_loader,
                                                                                                    desc="Testing"):
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(first_device), attention_mask.to(
                first_device), token_type_ids.to(first_device), labels.to(first_device)
            audio_feature, audio_mask = audio_feature.to(first_device), audio_mask.to(first_device)
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                audio_feature=audio_feature, audio_mask=audio_mask, labels=labels)
            lengths = torch.sum(attention_mask, 1)
            for i, length in enumerate(lengths):
                y_true.append(labels[i][0:length - 2])
                y_pred.append(logits[i][0:length - 2])

            p, r, f = compute_scores(y_pred, y_true, tag2index)

    print("\nP: %f, R: %f, F: %f" % (p, r, f))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_data_path",
                        default=None,
                        type=str,
                        help="The training data path.")
    parser.add_argument("--train_num",
                        default=50,
                        type=int,
                        help="The number of train_data.")
    parser.add_argument("--eval_data_path",
                        default=None,
                        type=str,
                        help="The eval/testing data path.")
    parser.add_argument("--audio_encoder_config_path",
                        default=None,
                        type=str,
                        help="The config path of audio_encoder.")

    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--eval_model", default=None, type=str,
                        help="")
    parser.add_argument("--num_multi_attention_layers",
                        default=3,
                        type=int,
                        help="The number of multi_attention_layers for multi-attention gating mechanism.")
    
    parser.add_argument("--attention_probs_dropout_prob",
                        default=0.1,
                        type=float,
                        help="The attention_probs_dropout_prob for multi-attention gating mechanism.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=30,
                        help="random seed for initialization")
    parser.add_argument('--first_device_id',
                        type=int,
                        default=0,
                        help="First device id to perform")

    parser.add_argument('--second_device_id',
                        type=int,
                        default=1,
                        help="Second device id to perform multi-attention gating mechanism. ")

    parser.add_argument('--model_name', type=str, default=None, help="")

    args = parser.parse_args()
    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    else:
        raise ValueError('At least one of `do_train`, `do_eval`, `do_predict` must be True.')


if __name__ == "__main__":
    main()
