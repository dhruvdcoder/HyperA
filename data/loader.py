from hypernn import config
import torchtext.data as data
import torchtext
#import hypernn.embeddings as pretrained_embs
import torch
import logging
logger = logging.getLogger(__name__)


def prepare_multiNLI(return_test_set=False,
                     embs_file=config.default_poincare_glove,
                     max_seq_len=config.cmd_args.max_seq_len,
                     batch_size=128,
                     device=torch.device('cpu'),
                     emb_cache_file=config.cmd_args.vector_cache,
                     use_pretrained=config.cmd_args.use_pretrained):
    logger.info('Loading MultiNLI data')
    logger.info('Using max_seq_len ={}'.format(max_seq_len))
    logger.info('Batch_size: {}'.format(batch_size))
    # prep the field types
    inputs = data.Field(lower=True, tokenize='spacy', batch_first=True, fix_length=max_seq_len)

    answers = data.LabelField(batch_first=True)

    test_set = config.cmd_args.test_set
    # load and build vocab for dataset
    logger.info('Loading data from {}'.format(config.data_dir / 'multinli'))
    train, dev, test = torchtext.datasets.MultiNLI.splits(
        inputs,
        answers,
        root=str((config.data_dir).absolute()),
        train=config.cmd_args.train_set,
        validation=config.cmd_args.val_set,
        test=test_set)
    logger.info('Building vocab for data')
    inputs.build_vocab(train, dev, test)
    answers.build_vocab(train)
    # load pretrained embs
    if use_pretrained:
        logger.info('Loading pretrained embs')
        if emb_cache_file is not None:  # specified through cmd
            if emb_cache_file.is_file():
                logger.info('Loading embs from {}'.format(emb_cache_file))
                inputs.vocab.vectors = torch.load(emb_cache_file)
        else:
            # cache not specified through cmd
            # try loading from default cache
            # useful when resuming training
            if config.emb_cache_file.is_file():
                logger.info('Loading embs from {}'.format(
                    config.emb_cache_file))
                inputs.vocab.vectors = torch.load(config.emb_cache_file)
            else:  # load pretrained emb from main emb file
                logger.info('Loading embs from {}'.format(embs_file))
                vectors = torchtext.vocab.Vectors(embs_file)
                inputs.vocab.set_vectors(vectors.stoi, vectors.vectors,
                                         vectors.dim)
                # save to default cache
                logger.info('Saving emb cache to {}'.format(
                    config.emb_cache_file))
                torch.save(inputs.vocab.vectors, config.emb_cache_file)
    else:
        # dont do anything. .vector of vocab will be None.
        # this will be used by the Models to tell HyperEmbeddings
        # to do appropriate initialization
        logger.info("Not loading any pretrained embs...")
        pass

    if not return_test_set:
        test = None
        test_iter = None  # free memory
        train_iter, dev_iter = data.BucketIterator.splits(
            (train, dev), batch_size=batch_size, device=device)
    else:
        train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=batch_size, device=device)

    if return_test_set == True:
        return (train_iter, dev_iter, test_iter), inputs,answers
    else:
        return (train_iter, dev_iter, test_iter), inputs



def get_vocab(load_test_set):
    pass


if __name__ == '__main__':
    inputs = data.Field(lower=True, tokenize='spacy', batch_first=True)
    answers = data.Field(sequential=False, batch_first=True)
    train, dev, test = torchtext.datasets.MultiNLI.splits(
        inputs, answers, root=str(config.data_dir.absolute()))
    inputs.build_vocab(train, dev, test)
