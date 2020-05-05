import argparse
import gzip
import logging
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn

import batch_generator
import corpus
import data
import evaluator
import lm_trainer
import model
import model_wrapper
import splitcross
import utils
import validator

logger = logging.getLogger('train_lm')

def mk_corpus(args):
    import hashlib
    corpus_args = '{}{}{}{}{}{}{}'.format(args.data,args.start_symbol,args.min_counts,args.max_vocab_size,args.unkify,args.preproc,args.reverse,args.add_document_start)
    fn = 'cached_corpus/corpus.{}.data'.format(hashlib.md5(corpus_args.encode()).hexdigest())
    if os.path.exists(fn) and args.use_cache:
        print('Loading cached dataset...')
        corpus_ = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus_ = corpus.Corpus(
            args.mode,
            args.add_document_start,
            args.data,
            args.start_symbol,
            args.min_counts,
            args.max_vocab_size,
            args.unkify,
            args.preproc,
            args.reverse,
            args.min_length,
            args.max_unk_ratio)

        if args.use_cache:
            if not os.path.exists('cached_corpus'):
                os.makedirs('cached_corpus', exist_ok=True)
            torch.save(corpus_, fn)

    logger.info('First sentence: {}'.format(corpus_.train_sentences[0]))
    logger.info('Ids: {}'.format(corpus_.train_tensors[0]))

    if args.output_processed_corpus:
        def process_one(tensors, prefix):
            vocab = corpus_.vocab
            fn = os.path.join(args.data, '{}.processed.txt.gz'.format(prefix))
            with gzip.open(fn, 'wt') as f:
                for tensor in tensors:
                    f.write(' '.join(vocab.value(w) for w in tensor) + '\n')
            print('Wrote preprocessed {} sentences in {}'.format(prefix, fn))
        process_one(corpus_.train_tensors, 'train')
        process_one(corpus_.val_tensors, 'valid')
        process_one(corpus_.test_tensors, 'test')
        # tensors = corpus_.train_tensors
        # vocab = corpus_.vocab
        # fn = os.path.join(args.data, 'train.processed.txt.gz')
        # with gzip.open(fn, 'wt') as f:
        #     for tensor in tensors:
        #         f.write(' '.join(vocab.value(w) for w in tensor) + '\n')
        # print('Wrote preprocessed train sentences in {}'.format(fn))

    return corpus_

def mk_tag_corpus(args, word_corpus):
    word_vocab = word_corpus.vocab

    corpus_ = corpus.TagCorpus(
        args.tag_data, args.start_symbol, word_vocab, args.tag_min_words,
        args.fix_word_vocab, args.build_tag_vocab, args.reverse)

    logger.info('First tagged sentence: {}'.format(corpus_.train_sentences[0]))
    logger.info('Ids: {}'.format(corpus_.train_tensors[0]))

    return corpus_

def load_model(fn, device=None):
    with open(fn, 'rb') as f:
        modelwrap, optimizer = torch.load(f) if device is None else torch.load(f, map_location=device)
        assert isinstance(modelwrap, model_wrapper.RNNModelWrapper)
    return modelwrap, optimizer

def load_for_resume(args):
    logger.info('Resuming model ...')
    modelwrap, optimizer = load_model(args.resume)

    if args.amp:
        from apex import amp
        modelwrap, optimizer = amp.initialize(modelwrap, optimizer, opt_level='O1')
    rnn = modelwrap.rnn
    assert isinstance(modelwrap, model_wrapper.RNNModelWrapper)

    optimizer.param_groups[0]['lr'] = args.lr
    rnn.dropouti, rnn.dropouth, rnn.dropout, rnn.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for subnet in rnn.rnns:
            if type(subnet) == WeightDrop: subnet.dropout = args.wdrop
            elif subnet.zoneout > 0: subnet.zoneout = args.wdrop
    return modelwrap, optimizer

def mk_model(args, vocab, device, tag_vocab=None):
    rnn = model.RNNModel(args.model, vocab, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, args.lock_drop, padding_idx = vocab.index(data.PAD))

    optimizer = None
    if args.resume:
        rnn, optimizer = load_for_resume(args)
        vocab = rnn.vocab

    if args.mode == 'multitask':
        if args.build_tag_vocab:
            modelwrap = model_wrapper.PartiallySharedMultiTaskModel(rnn, tag_vocab)
        else:
            modelwrap = model_wrapper.FullySharedMultiTaskModel(rnn)
    else:
        do_bc = args.neg_criterion == 'binary-prediction'
        modelwrap = model_wrapper.RNNModelWrapper(rnn,
                                                  do_binary_classification=do_bc)
    modelwrap.to(device)

    return modelwrap, optimizer

def mk_config(args):
    if args.plateau_lr_decay:
        assert args.start_lr_decay < 0

    return lm_trainer.TrainConfig(
        args.lr, args.wdecay, args.when, args.optimizer, args.nonmono, args.clip,
        args.alpha, args.beta, args.start_lr_decay, args.plateau_lr_decay,
        args.start_plateau_lr_decay, args.plateau_patience, args.lr_decay_gamma,
        args.non_average, args.agreement_loss_alpha, args.amp,
        args.do_sample_average_before, args.within_batch_neg_prob, args.normalize_negative_loss,
        args.sample_average_jointly)

def run_epochs(trainer, batch_gen, val_data, log_interval, validate_batches, epochs):
    try:
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            trainer.run_epoch(batch_gen, val_data, log_interval, validate_batches, epoch)
    except KeyboardInterrupt:
        logger.info('-' * 89)
        logger.info('Exiting from training early')

def final_eval(evaluator, test_gen, save):
    if os.path.exists(save):
        rnn, _ = load_model(save)
    test_loss, avg_loss, margin_loss, accuracy = evaluator.evaluate(rnn, test_gen)
    logger.info('=' * 89)
    logger.info('| End of training | test loss {:5.2f} | '
                'agreement loss {:5.2f} | accuracy {:.5f} | '
                'test ppl {:8.2f} | test bpc {:8.3f}'.format(
                    test_loss,
                    margin_loss,
                    accuracy,
                    math.exp(avg_loss),
                    avg_loss / math.log(2)))
    logger.info('=' * 89)


def train_document_model(args, device):
    corpus = mk_corpus(args)
    vocab = corpus.vocab

    evaluator_ = evaluator.DocumentEvaluator(args.bptt, vocab.index(data.PAD))
    validator_ = validator.Validator(evaluator_, args.save)

    model, optimizer = mk_model(args, vocab, device)
    batch_gen = batch_generator.DocumentBatchGenerator(
        corpus.train_tensors, args.batch_size, args.bptt, args.shuffle)
    val_gen = batch_generator.DocumentBatchGenerator(corpus.val_tensors, 10, args.bptt)
    test_gen = batch_generator.DocumentBatchGenerator(corpus.test_tensors, 1, args.bptt)

    trainer = lm_trainer.DocumentLMTrainer(
        model, validator_, mk_config(args), args.bptt)

    if optimizer:
        trainer.optimizer = optimizer

    run_epochs(trainer, batch_gen, val_gen, args.log_interval, args.validate_batches, args.epochs)
    final_eval(validator_.evaluator, test_gen, args.save)


def train_document_agreement_model(args, device):
    corpus = mk_corpus(args)
    vocab = corpus.vocab

    evaluator_ = evaluator.DocumentEvaluator(args.bptt, vocab.index(data.PAD))
    validator_ = validator.Validator(evaluator_, args.save)

    model, optimizer = mk_model(args, vocab, device)
    batch_gen = batch_generator.DocumentBatchGenerator(
        corpus.train_tensors, args.batch_size, args.bptt, args.shuffle)
    val_gen = batch_generator.DocumentBatchGenerator(corpus.val_tensors, 10, args.bptt)
    test_gen = batch_generator.DocumentBatchGenerator(corpus.test_tensors, 1, args.bptt)

    trainer = lm_trainer.DocumentLMTrainer(
        model, validator_, mk_config(args), args.bptt)

    if optimizer:
        trainer.optimizer = optimizer

    run_epochs(trainer, batch_gen, val_gen, args.log_interval, args.validate_batches, args.epochs)
    final_eval(validator_.evaluator, test_gen, args.save)


def train_sentence_model(args, device):
    corpus = mk_corpus(args)
    vocab = corpus.vocab
    pad_id = vocab.index(data.PAD)

    evaluator_ = evaluator.SentenceEvaluator(pad_id)
    validator_ = validator.Validator(evaluator_, args.save)

    model, optimizer = mk_model(args, vocab, device)
    batch_gen = batch_generator.SentenceBatchGenerator(
        corpus.train_tensors, args.batch_size, pad_id, args.shuffle, args.length_bucket)

    if args.validate_agreement:
        _, val_tensor_pairs = corpus.mk_agreement_pairs(False, args.target_syntax)
        val_gen = batch_generator.SentenceAndAgreementBatchGenerator(
            corpus.val_tensors, val_tensor_pairs, 100, pad_id)
    else:
        val_gen = batch_generator.SentenceBatchGenerator(corpus.val_tensors, 100, pad_id)
    test_gen = batch_generator.SentenceBatchGenerator(corpus.test_tensors, 100, pad_id)

    trainer = lm_trainer.SentenceLMTrainer(model, validator_, mk_config(args))

    if optimizer:
        trainer.optimizer = optimizer

    run_epochs(trainer, batch_gen, val_gen, args.log_interval, args.validate_batches, args.epochs)
    final_eval(validator_.evaluator, test_gen, args.save)

def train_multitask_model(args, device):
    corpus = mk_corpus(args)

    tag_corpus = mk_tag_corpus(args, corpus)
    vocab = tag_corpus.word_vocab # might be updated
    tag_vocab = tag_corpus.tag_vocab

    pad_id = vocab.index(data.PAD)
    tag_pad_id = tag_vocab.index(data.PAD)

    evaluator_ = evaluator.SentenceEvaluator(pad_id)
    validator_ = validator.Validator(evaluator_, args.save)

    model, optimizer = mk_model(args, vocab, device, tag_vocab)

    batch_gen = batch_generator.SentenceAndTagBatchGenerator(
        corpus.train_tensors, tag_corpus.train_tensors, args.batch_size, pad_id, tag_pad_id,
        args.shuffle, args.length_bucket, args.upsample_tags, args.tag_loss_alpha)

    val_gen = batch_generator.SentenceBatchGenerator(corpus.val_tensors, 100, pad_id)
    test_gen = batch_generator.SentenceBatchGenerator(corpus.test_tensors, 100, pad_id)

    trainer = lm_trainer.SentenceLMTrainer(model, validator_, mk_config(args))

    if optimizer:
        trainer.optimizer = optimizer

    run_epochs(trainer, batch_gen, val_gen, args.log_interval, args.validate_batches, args.epochs)
    final_eval(validator_.evaluator, test_gen, args.save)

def train_agreement_model(args, device):
    # vocab is shared with the pretrained (sentence) model
    # TODO: do we truly need to use load_for_resume instead of load_model?
    # model, optimizer = load_for_resume(args)
    model, optimizer = load_model(args.resume)

    vocab = model.vocab
    pad_id = vocab.index(data.PAD)

    def read_sentences(path):
        return data.read_sentences(
            os.path.join(args.data, path), False, False, args.start_symbol)

    train_sentences = read_sentences('train.txt')
    val_sentences = read_sentences('valid.txt')
    test_sentences = read_sentences('test.txt')

    val_tensors = data.to_tensors(val_sentences, vocab)
    test_tensors = data.to_tensors(test_sentences, vocab)

    negative_examples = data.read_negative_examples(
        os.path.join(args.data, 'negative_agreements.train.txt.gz'))
    agreement_pairs = data.mk_agreement_pairs(
        train_sentences, negative_examples, args.ignore_simple_agreement)

    # TODO: maybe consider args.max_unk_ratio and min_length here
    tensor_pairs = data.to_tensor_pairs(agreement_pairs, vocab)

    # TODO: halfing batch_size might be better.
    batch_gen = batch_generator.AgreementPairBatchGenerator(
        tensor_pairs, args.batch_size, pad_id, args.shuffle)
    val_gen = batch_generator.SentenceBatchGenerator(val_tensors, 100, pad_id)
    test_gen = batch_generator.SentenceBatchGenerator(test_tensors, 100, pad_id)

    evaluator_ = evaluator.SentenceEvaluator(pad_id)
    validator_ = validator.Validator(evaluator_, args.save)

    neg_calc = lm_trainer.SentenceMarginLossCalculator(args.margin)
    trainer = lm_trainer.SentenceWithAgreementSentenceLossTrainer(
        model, validator_, mk_config(args), neg_calc)

    # TODO: it may be better to reset the optimizer.
    if args.reset_optimizer:
        params = list(model.parameters())
        lr = args.lr
        weight_decay = args.wdecay
        optimizer = torch.optim.ASGD(model.rnn.parameters(), lr=lr, weight_decay=weight_decay, t0=0, lambd=0.)
    trainer.optimizer = optimizer

    run_epochs(trainer, batch_gen, val_gen, args.log_interval, args.validate_batches, args.epochs)
    final_eval(validator_.evaluator, test_gen, args.save)

def train_sentence_and_agreement_model(args, device):
    corpus = mk_corpus(args)
    vocab = corpus.vocab
    pad_id = vocab.index(data.PAD)

    train_tensor_pairs, val_tensor_pairs = corpus.mk_agreement_pairs(
        args.ignore_simple_agreement, args.target_syntax, args.margin_prefix_only)

    if args.neg_criterion == 'binary-prediction':
        assert args.margin_prefix_only
        train_tensor_pairs = utils.transform_agreement_pairs_to_final_binary_prediction(
            train_tensor_pairs, vocab)
        val_tensor_pairs = utils.transform_agreement_pairs_to_final_binary_prediction(
            val_tensor_pairs, vocab)

    if args.neg_criterion == 'unlikelihood-sentence':
        neg_calc = lm_trainer.SentenceUnlikelihoodLossCalculator()
    elif args.neg_criterion == 'unlikelihood':
        neg_calc = lm_trainer.SentenceLastTokenUnlikelihoodLossCalculator()
    elif args.neg_criterion == 'neg-likelihood':
        neg_calc = lm_trainer.SentenceWrongLikelihoodLossCalculator()
    elif args.neg_criterion == 'margin':
        neg_calc = lm_trainer.SentenceMarginLossCalculator(args.margin)
    elif args.neg_criterion == 'binary-prediction':
        neg_calc = lm_trainer.SentenceLastTokenBinaryPredictionLossCalculator()

    evaluator_ = evaluator.SentenceEvaluator(pad_id, neg_calc)
    validator_ = validator.Validator(evaluator_, args.save)

    model, optimizer = mk_model(args, vocab, device)
    batch_gen = batch_generator.SentenceAndAgreementBatchGenerator(
        corpus.train_tensors, train_tensor_pairs, args.batch_size, pad_id, args.shuffle,
        args.length_bucket, args.upsample_agreement, args.agreement_loss_alpha,
        half_agreement_batch = args.half_agreement_batch,
        one_pair_per_sent = args.one_pair_per_sent,
        agreement_sample_ratio = args.agreement_sample_ratio,
        prefer_obj_rel = args.margin_prefer_obj_rel)

    # whether to include agreement batches in validation is decided here.
    if args.validate_agreement:
        val_gen = batch_generator.SentenceAndAgreementBatchGenerator(
            corpus.val_tensors, val_tensor_pairs, 100, pad_id)
    else:
        val_gen = batch_generator.SentenceBatchGenerator(corpus.val_tensors, 100, pad_id)
    test_gen = batch_generator.SentenceBatchGenerator(corpus.test_tensors, 100, pad_id)

    trainer = lm_trainer.SentenceWithAgreementSentenceLossTrainer(
        model, validator_, mk_config(args), neg_calc)

    if optimizer:
        trainer.optimizer = optimizer

    run_epochs(trainer, batch_gen, val_gen, args.log_interval, args.validate_batches, args.epochs)
    final_eval(validator_.evaluator, test_gen, args.save)


def train_sentence_with_token_agreement_model(args, device):
    corpus = mk_corpus(args)
    vocab = corpus.vocab
    pad_id = vocab.index(data.PAD)

    if args.neg_criterion == 'binary-prediction':
        train_neg_tokens, val_neg_tokens = corpus.mk_agreement_tokens_for_binary_prediction(
            args.ignore_simple_agreement, args.target_syntax)
    else:
        train_neg_tokens, val_neg_tokens = corpus.mk_agreement_tokens(
            args.ignore_simple_agreement, args.target_syntax, args.exclude_targets, args.only_targets)

    if args.neg_criterion == 'unlikelihood':
        neg_calc = lm_trainer.TokenUnlikelihoodLossCalculator()
    elif args.neg_criterion == 'margin':
        neg_calc = lm_trainer.TokenMarginLossCalculator(args.margin)
    elif args.neg_criterion == 'binary-prediction':
        neg_calc = lm_trainer.TokenBinaryPredictionLossCalculator()
    elif args.neg_criterion == 'sent-margin-within':
        if args.do_sample_average_before == False:
            logger.info('do_sample_average_before should be True for sent-margin-within. Changed to True.')
            args.do_sample_average_before = True
        neg_calc = lm_trainer.WithinBatchSentenceMarginLossCalculator(args.margin)
    elif args.neg_criterion == 'sent-unlikelihood-within':
        if args.do_sample_average_before == False:
            logger.info('do_sample_average_before should be True for sent-unlikelihood-within. Changed to True.')
            args.do_sample_average_before = True
        neg_calc = lm_trainer.WithinBatchSentenceUnlikelihoodLossCalculator()

    evaluator_ = evaluator.SentenceEvaluator(pad_id, neg_calc)
    validator_ = validator.Validator(evaluator_, args.save)

    model, optimizer = mk_model(args, vocab, device)
    batch_gen = batch_generator.SentenceWithAgreementBatchGenerator(
        corpus.train_tensors, train_neg_tokens, args.batch_size, pad_id, args.shuffle,
        args.length_bucket)

    if args.validate_agreement:
        logger.info('Consider agreement loss for decising early stopping.')
        val_gen = batch_generator.SentenceWithAgreementBatchGenerator(
            corpus.val_tensors, val_neg_tokens, 100, pad_id)
    else:
        val_gen = batch_generator.SentenceBatchGenerator(corpus.val_tensors, 100, pad_id)
    test_gen = batch_generator.SentenceBatchGenerator(corpus.test_tensors, 100, pad_id)

    trainer = lm_trainer.SentenceWithAgreementTokenLossTrainer(
        model, validator_, mk_config(args), neg_calc)

    if optimizer:
        trainer.optimizer = optimizer

    run_epochs(trainer, batch_gen, val_gen, args.log_interval, args.validate_batches, args.epochs)
    final_eval(validator_.evaluator, test_gen, args.save)


def run_train(args):
    if args.gpu and args.gpu >= 0:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cuda:{}".format(utils.find_idle_gpu()) if torch.cuda.is_available() else "cpu")

    logger.info("device: {}".format(device))

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.mode == 'document':
        train_document_model(args, device)
    elif args.mode == 'sentence':
        train_sentence_model(args, device)
    elif args.mode == 'multitask':
        train_multitask_model(args, device)
    elif args.mode == 'agreement':
        train_agreement_model(args, device)
    elif args.mode == 'sentagree':
        if args.neg_mode == 'mtl':
            train_sentence_and_agreement_model(args, device)
        elif args.neg_mode == 'token':
            train_sentence_with_token_agreement_model(args, device)
    else:
        raise ValueError('unknown --mode: {}.'.format(args.mode))


def main():
    parser = argparse.ArgumentParser('Train Multi-task RNN/LSTM Language Models')
    parser.add_argument('--mode', default='document', choices=['document', 'sentence', 'multitask', 'agreement', 'sentagree'])
    # parser.add_argument('--network', default='full_share', choices=['full_share', 'partial_share'])

    parser.add_argument('--reverse', action='store_true',
                        help='Training with reversed sentences')
    parser.add_argument('--add-document-start', action='store_true',
                        help='put <eos> or <bos> (depending on --start-symbol) at the begin of a document')
    parser.add_argument('--start-symbol', default='<eos>', choices=['<eos>', '<bos>'])
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--length-bucket', action='store_true', help='If true, each mini-batch is created from larger buckets segmented by sentence lengths to make sentences in a batch have similar lengths.')
    parser.add_argument('--unkify', default='unk',
                        choices=['choe_charniak', 'pos', 'unk'])
    parser.add_argument('--preproc', nargs='*', choices=['lower', 'delnum'], default=[])
    parser.add_argument('--min-counts', type=int, default=1)
    parser.add_argument('--max-vocab-size', type=int, default=-1)
    parser.add_argument('--min-length', type=int, default=1)
    parser.add_argument('--max-unk-ratio', type=float, default=0.0,
                        help='Discard a sentence if the ratio of unk tokens exceeds this. (0 for accepting all.)')

    parser.add_argument('--amp', action='store_true')

    parser.add_argument('--output-processed-corpus', action='store_true')

    # for agreement
    parser.add_argument('--target-syntax', nargs='*', choices=['agreement', 'reflexive'], default=['agreement'])
    parser.add_argument('--ignore-simple-agreement', action='store_true')
    parser.add_argument('--margin', type=float, default=3.0)
    parser.add_argument('--reset-optimizer', action='store_true')
    parser.add_argument('--upsample-agreement', action='store_true', help='For joint training.')
    parser.add_argument('--agreement-loss-alpha', type=float, default=1.0)
    parser.add_argument('--half-agreement-batch', action='store_true',
                        help='If True, make the size of agreement batch as half')
    parser.add_argument('--one-pair-per-sent', action='store_true',
                        help='If True, only one agreement pair is generated for each sent.')
    parser.add_argument('--agreement-sample-ratio', type=float, default=0.5,
                        help='If > 0, # batches for agreement becomes this ratio against the sentence batches; not compatible with --one-pair-per-sent and --upsample-agreement.')
    parser.add_argument('--validate-agreement', action='store_true',
                        help='If true, include agreement loss for early stopping.')
    parser.add_argument('--margin-prefix-only', action='store_true',
                        help='If true, margin loss is calculated only with prefix up to a target verb.')
    parser.add_argument('--margin-prefer-obj-rel', action='store_true',
                        help='If true, batches for margin training prefer obj-rel agreement examples (given by obj_rel_agreements.train.txt.gz).')

    # token-level negative loss
    parser.add_argument('--neg-mode', default='mtl',
                        choices=['mtl', 'token'])
    parser.add_argument('--neg-criterion', default='margin',
                        choices=['unlikelihood',
                                 'unlikelihood-sentence',
                                 'neg-likelihood',
                                 'margin',
                                 'binary-prediction',
                                 'sent-margin-within',
                                 'sent-unlikelihood-within'])

    parser.add_argument('--do-sample-average-before', action='store_true')
    parser.add_argument('--sample-average-jointly', action='store_true')
    parser.add_argument('--within-batch-neg-prob', type=float, default=-1.0)
    parser.add_argument('--normalize-negative-loss', action='store_true')

    parser.add_argument('--exclude-targets', nargs='*', type=str, default=[])
    parser.add_argument('--only-targets', nargs='*', type=str, default=[])

    # for multi-tasking
    parser.add_argument('--tag-min-words', type=int, default=1,
                        help='min-counts for words appeared in tag training data.')
    parser.add_argument('--fix-word-vocab', action='store_true',
                        help='If true, do not increase the size of vocab for words in tag corpus.')
    parser.add_argument('--build-tag-vocab', action='store_true',
                        help='Whether tag vocab is separated from the word vocab. If true, create a separate vocab.')
    parser.add_argument('--upsample-tags', action='store_true',
                        help='If true, do upsampling tag bathces, and alternate between sentence batch (main task) and tag batch (sub task).')
    parser.add_argument('--tag-loss-alpha', type=float, default=1.0)

    parser.add_argument('--use-cache', action='store_true')

    parser.add_argument('--data', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--tag-data', type=str,
                        help='tagged data used for multi-task learning.')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=18,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length for bptt (only for document mode)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.1,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.1,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.0,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--lock-drop', action='store_false')
    parser.add_argument('--wdrop', type=float, default=0.0,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=0,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=0,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--start-lr-decay', type=int, default=-1,
                        help='Which epoch to start LR decay. (-1 for never)')
    parser.add_argument('--plateau-lr-decay', action='store_true',
                        help='Reduce learning rate by a factor of --lr-decay-gamma when validation accuracy drops at a checkpoint.')
    parser.add_argument('--start-plateau-lr-decay', type=int, default=5,
                        help='Minimum epoch to be judged to start plateau-lr-decay')
    parser.add_argument('--plateau-patience', type=int, default=1)
    parser.add_argument('--lr-decay-gamma', type=float, default=0.5)
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--non-average', action='store_true',
                        help='In default, --optimizer=sgd means averaged SGD. This disables averaging.')
    parser.add_argument('--when', nargs="+", type=int, default=[-1],
                        help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    parser.add_argument('--validate-batches', type=int, default=5000,
                        help='If 0, validate every epoch; if not 0, validate every # batches')

    parser.add_argument('--tied', action='store_false')
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("{}.log".format(args.save)),
        logging.StreamHandler()
    ])

    logger.info('Args: {}'.format(args))

    run_train(args)

if __name__ == '__main__':
    main()
