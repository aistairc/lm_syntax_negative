
import math
import logging
import torch

import model_wrapper

logger = logging.getLogger('train_lm')

class Validator(object):

    def __init__(self, evaluator, save_path):
        self.evaluator = evaluator
        self.save_path = save_path

        self.best_val_loss = []
        self.stored_loss = 100000000

    def save(self, model, optimizer):
        assert isinstance(model, model_wrapper.RNNModelWrapper)
        with open(self.save_path, 'wb') as f:
            torch.save([model, optimizer], f)

    def save_epoch(self, model, optimizer, epoch):
        assert isinstance(model, model_wrapper.RNNModelWrapper)
        p = '{}.e{}'.format(self.save_path, epoch)
        with open(self.save_path, 'wb') as f:
            torch.save([model, optimizer], f)

    def validate(self, model, optimizer, val_gen, epoch):
        assert isinstance(model, model_wrapper.RNNModelWrapper)

        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.rnn.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2, avg_loss, margin_loss, accuracy = self.evaluator.evaluate(
                model, val_gen)

            logger.info('-' * 89)
            logger.info('| evaluate on valid | valid loss {:5.2f} | '
                        'agreement loss {:5.2f} | accuracy {:.5f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                            val_loss2,
                            margin_loss,
                            accuracy,
                            math.exp(avg_loss),
                            avg_loss / math.log(2)))
            logger.info('-' * 89)

            if val_loss2 < self.stored_loss:
                self.save(model, optimizer)
                logger.info('Saving Averaged!')
                self.stored_loss = val_loss2

            for prm in model.rnn.parameters():
                prm.data = tmp[prm].clone()

            return val_loss2

        else:
            # val_loss may contain loss for agreement. avg_loss is just LM loss.
            val_loss, avg_loss, margin_loss, accuracy = self.evaluator.evaluate(
                model, val_gen)
            logger.info('-' * 89)
            logger.info('| evaluate on valid | valid loss {:5.2f} | '
                        'agreement loss {:5.2f} | accuracy {:.5f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                            val_loss,
                            margin_loss,
                            accuracy,
                            math.exp(avg_loss),
                            avg_loss / math.log(2)))
            logger.info('-' * 89)

            if val_loss < self.stored_loss:
                self.save(model, optimizer)
                logger.info('Saving model (new best validation)')
                self.stored_loss = val_loss

            self.best_val_loss.append(val_loss)

            return val_loss
