
import torch

from lm_trainer import TokenNegLossCalculator
import utils

class Evaluator(object):

    def __init__(self, pad_id):
        self.pad_id = pad_id

class DocumentEvaluator(Evaluator):

    def __init__(self, bptt, pad_id):
        super().__init__(pad_id)
        self.bptt = bptt

    def evaluate(self, model, batch_gen):
        device = next(model.parameters()).device
        # Turn on evaluation mode which disables dropout.
        model.rnn.eval()
        total_loss = 0
        ntokens = model.vocab.size
        hidden = model.rnn.init_hidden(batch_gen.batch_size)
        for data, targets in batch_gen.for_eval():
            data = data.to(device)
            targets = targets.to(device)
            output, hidden = model.rnn(data, hidden)
            total_loss += data.size(1) * model.loss(output, targets).item()
            hidden = utils.repackage_hidden(hidden)
        return total_loss / batch_gen.data.size(1)

class SentenceEvaluator(Evaluator):

    def __init__(self, pad_id, neg_loss_calculator=None, exclude=[]):
        """exclude: excluded token index. exclude=[0, -1, -2] means excluding first,
        last tokens, and eos.
        """
        super().__init__(pad_id)
        if neg_loss_calculator:
            self.neg_loss_calculator = neg_loss_calculator

            # neg_loss_calculator should only be used during training.
            # exclude, on the other hand, should only be used for testing.
            assert len(exclude) == 0

        self.exclude = exclude

    def evaluate(self, model, batch_gen):
        with torch.no_grad():
            return self._evaluate(model, batch_gen)

    def _evaluate(self, model, batch_gen):
        device = next(model.parameters()).device
        model.eval()
        all_tokens = 0
        total_cross_entropy_loss = 0.0
        total_margin_loss = 0.0
        total_correct = 0
        total_margin_sents = 0
        for batch in batch_gen.for_eval():
            if len(batch) == 3 or len(batch) == 4:
                if len(batch) == 3:
                    sources, lengths, targets = batch
                    ent_loss = self._cross_entropy_loss(
                        model, sources, lengths, targets, device)
                else:
                    sources, lengths, targets, negatives = batch
                    ent_loss, margin_loss = self._cross_entropy_token_neg_loss(
                        model, sources, lengths, targets, negatives, device)
                    total_margin_loss += margin_loss

                if len(self.exclude) > 0:
                    nexclude = len(self.exclude)
                    non_zero = sum([max(0, l - nexclude) for l in lengths]).item()
                else:
                    non_zero = lengths.sum().item()

                all_tokens += non_zero
                total_cross_entropy_loss += ent_loss * non_zero
            else:
                margin_loss, _, _ = self.neg_loss_calculator.loss(model, batch)
                total_margin_loss += margin_loss.item() * len(batch[0])
                # margin_loss, n_correct = self._margin_loss(model, batch, device)
                # total_margin_loss += margin_loss * len(batch[0])
                # total_correct += n_correct
                # total_margin_sents += len(batch[0])

        total_loss = total_cross_entropy_loss + total_margin_loss
        avg_loss = total_cross_entropy_loss / all_tokens
        agreement_accuracy = total_correct / total_margin_sents \
                             if total_margin_sents > 0 else 0
        return total_loss, avg_loss, total_margin_loss, agreement_accuracy

    def _cross_entropy_loss(self, model, sources, lengths, targets, device):
        sources = sources.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        output, _ = model.rnn(sources, input_lengths = lengths)

        targets = self._mask_targets_by_exlude(targets, lengths)

        return model.loss(output, targets).item()

    def _cross_entropy_token_neg_loss(
            self, model, sources, lengths, targets, negatives, device):
        assert isinstance(self.neg_loss_calculator, TokenNegLossCalculator)
        sources = sources.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        output, _ = model.rnn(sources, input_lengths = lengths)

        raw_loss = model.loss(output, targets, reduce=False)

        unlikelihood_loss = self.neg_loss_calculator.loss(
            model, output, targets, negatives, sources.size(1))

        unlikelihood_sum = unlikelihood_loss.sum()
        ntokens = lengths.sum()
        ent_avg = raw_loss.sum() / ntokens
        return (ent_avg.item(), unlikelihood_sum.item())

    def _mask_targets_by_exlude(self, targets, lengths):
        if len(self.exclude) == 0:
            return targets
        maxlen = lengths.max()
        offset = 0
        for l in lengths:
            for e in self.exclude:
                if e >= 0:
                    idx = offset + e
                else:
                    idx = offset + l + e
                targets[idx] = self.pad_id
            offset += maxlen
        return targets

    # def _margin_loss(self, model, batch, device):
    #     assert self.margin_criterion is not None
    #     gold_srcs, lengths, gold_tgts, wrong_srcs, wrong_tgts = batch
    #     gold_srcs = gold_srcs.to(device)
    #     lengths = lengths.to(device)
    #     gold_tgts = gold_tgts.to(device)
    #     wrong_srcs = wrong_srcs.to(device)
    #     wrong_tgts = wrong_tgts.to(device)

    #     gold_outputs, _ = model.rnn(gold_srcs, input_lengths = lengths, return_h = False)
    #     wrong_outputs, _ = model.rnn(wrong_srcs, input_lengths = lengths, return_h = False)

    #     gold_probs = model.loss(gold_outputs, gold_tgts, reduce=False) * -1.0 # (total_length, 1)
    #     wrong_probs = model.loss(wrong_outputs, wrong_tgts, reduce=False) * -1.0

    #     # We first decompose *_probs into sentences using `lengths`.
    #     gold_sent_probs = gold_probs.new_zeros(lengths.size(0))
    #     wrong_sent_probs = gold_probs.new_zeros(lengths.size(0))
    #     offset = 0
    #     for i, l in enumerate(lengths):
    #         gold_sent_probs[i] = gold_probs[offset:offset+l].sum()
    #         wrong_sent_probs[i] = wrong_probs[offset:offset+l].sum()
    #         offset += l
    #     assert offset == lengths.sum()

    #     gold_sent_probs = gold_sent_probs.unsqueeze(1)
    #     wrong_sent_probs = wrong_sent_probs.unsqueeze(1)
    #     probs = torch.cat((gold_sent_probs, wrong_sent_probs), 1)
    #     targets = gold_sent_probs.new_zeros(gold_sent_probs.size(0), dtype=torch.long)

    #     n_correct = len([p for p in probs if p[0] > p[1]])

    #     return self.margin_criterion(probs, targets).item(), n_correct

    def word_stats(self, model, tensors, pad_id = None, calc_entropy = True):
        '''Given a list of sentences (word ids), calculate two stats: surprisals and entropys.

        Input tensors are *unsorted* sentences. This internally sorts the input first, for
        batch processing. After obtaining stats for sorted inputs, then, this returns the
        stats for each sentence after recovering the original sentence order.
        '''

        if not pad_id:
            pad_id = self.pad_id

        device = next(model.parameters()).device
        # batch_size = 100
        model.eval()

        sources, lengths, targets, perm_idx, reverse_idx = (
            self._sources_to_sorted_tensors(tensors, device, pad_id))

        output, _ = model.rnn(sources, input_lengths = lengths)
        output = output.detach()

        surps = model.loss(output, targets, reduce=False).detach()
        surps = surps.squeeze(1)
        entropys = model.entropy(output).detach() if calc_entropy else None

        list_surps = []
        list_ents = []

        offset = 0
        for j in range(len(lengths)):
            idx = j
            l = lengths[j].item()
            sent_surps = surps[offset:offset+l].cpu().numpy()
            list_surps.append(sent_surps)

            if calc_entropy:
                sent_ents = entropys[offset:offset+l].cpu().numpy()
                list_ents.append(sent_ents)
            offset += l

        list_surps = [list_surps[i] for i in reverse_idx]
        if calc_entropy:
            list_ents = [list_ents[i] for i in reverse_idx]
        else:
            list_ents = list_surps

        return (list_surps, list_ents)

    def _sources_to_sorted_tensors(self, tensors, device, pad_id):
        lengths = torch.tensor([len(s)-1 for s in tensors])
        lengths, perm_idx = lengths.sort(0, descending=True)

        sources = [tensors[i] for i in perm_idx]
        sources, lengths, targets = utils.get_sorted_sentences_batch(
                sources, 0, len(tensors), pad_id, sorted=True)

        sources = sources.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        reverse_idx = [0] * len(perm_idx)
        for i, order in enumerate(perm_idx):
            reverse_idx[order] = i

        return sources, lengths, targets, perm_idx, reverse_idx

    def get_layer_hiddens(self,
                          model,
                          tensors,
                          pad_id = None,
                          only_top_layer = False,
                          context = [0]):  # 0 means w_t, -1 means w_{t-1}
        """Run RNN on the in
        puts, and obtain a sequence of numpy arrays of the size
        [ndarray(num_layers, sequence_length, representation_dim)].
        """

        if not pad_id:
            pad_id = self.pad_id

        device = next(model.parameters()).device
        # batch_size = 100
        model.eval()

        sources, lengths, targets, perm_idx, reverse_idx = (
            self._sources_to_sorted_tensors(tensors, device, pad_id))

        _, _, outputs, _ = model.rnn(sources, input_lengths = lengths, return_h=True)
        # outputs is a list of tensors. Each tensor corresponds to one layer, and has
        # the size of (seq_len, batch_size, hidden_size).
        # (batch_size, seq_len, hidden_size), probably?
        outputs = [output.detach() for output in outputs]

        hidden_arrays = []

        # Single element list means that the only layer is the top layer.
        layers = [len(outputs)-1] if only_top_layer else list(range(len(outputs)))
        max_layer_dim = max([outputs[layer].size(2) for layer in layers])

        for i, length in enumerate(lengths):

            def layer_to_array(layer):
                context_array = []

                current_emb_dim = outputs[layer].size(2)
                if current_emb_dim < max_layer_dim:
                    remain_size = max_layer_dim - current_emb_dim
                    # Our LSTMs may have different hidden dim for each layer.
                    # This method makes dimensions of all layers the same, by filling 0
                    # for smaller dim layer.
                    def may_expand(emb):
                        return torch.cat([emb, emb.new_zeros(emb.size(0), remain_size)], 1)
                else:
                    # If dim of this layer is maximum, do nothing.
                    def may_expand(emb):
                        return emb

                for c in context:
                    # "1+c:1+c+length-1" means that index 1 is offset. Index 1
                    # corresponds to encoding of the first token (0 corresponds to BOS).
                    layer_output = outputs[layer][i,1+c:1+c+length-1,:]
                    layer_output = may_expand(layer_output)
                    context_array.append(layer_output)
                array = torch.cat(context_array, -1).unsqueeze(0)
                return array

            arrays = [layer_to_array(layer) for layer in layers]
            array = torch.cat(arrays, 0).cpu().numpy()
            # array = torch.cat(arrays, 0)
            # print(array.size())
            # array = array.cpu().numpy()
            hidden_arrays.append(array)

        hidden_arrays = [hidden_arrays[i] for i in reverse_idx]
        return hidden_arrays

