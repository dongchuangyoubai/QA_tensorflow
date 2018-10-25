import numpy as np

class squad_dataset(object):
    def __init__(self, question_file, context_file, answer_file):
        """
        Args:
            filename: path to the files
        """
        self.question_file = question_file
        self.context_file = context_file
        self.answer_file = answer_file

        self.length = None

    def iter_file(self, filename):
        print(filename)
        with open(filename) as f:
            for line in f:
                line = line.strip().split(" ")
                line = map(lambda tok: int(tok), line)
                yield line

    def __iter__(self):
        niter = 0

        question_file_iter = self.iter_file(self.question_file)
        answer_file_iter = self.iter_file(self.answer_file)
        context_file_iter = self.iter_file(self.context_file)

        for question, context, answer in zip(question_file_iter, context_file_iter, answer_file_iter):
            yield (question, context, answer)



    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length



def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return np.array(sequence_padded), np.array(sequence_length)


def pad_sequences(sequences, pad_tok):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    max_length = max([len(list(x)) for x in sequences])
    sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (question, context, answer) tuples
        minibatch_size: (int)
    Returns: 
        list of tuples
    """
    question_batch, context_batch, answer_batch = [], [], []

    for (q, c, a) in data:
        if len(question_batch) == minibatch_size:
            yield question_batch, context_batch, answer_batch
            question_batch, context_batch, answer_batch = [], [], []
        
        question_batch.append(q)
        context_batch.append(c)
        answer_batch.append(a)

    if len(question_batch) != 0:
        yield question_batch, context_batch, answer_batch



def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        nmatrix of embeddings (np array)
    """
    return np.load(filename)["glove"]

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        print(type(rev_vocab[0]))
        rev_vocab = [str(line).strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)
