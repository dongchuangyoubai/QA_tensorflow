import tensorflow as tf

from scripts.qa_model import Encoder, QASystem, Decoder
from scripts.config import Config
from scripts.data_utils import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_func():
    config = Config()
    train = squad_dataset(config.question_train, config.context_train, config.answer_train)
    dev = squad_dataset(config.question_dev, config.context_dev, config.answer_dev)

    embed_path = config.embed_path
    vocab_path = config.vocab_path
    vocab, rev_vocab = initialize_vocab(vocab_path)

    embeddings = get_trimmed_glove_vectors(embed_path)

    encoder = Encoder(config.hidden_size, config.num_layers)
    decoder = Decoder(config.hidden_size, config.num_layers)

    qa = QASystem(encoder, decoder, embeddings, config)

    with tf.Session() as sess:
        # ====== Load a pretrained model if it exists or create a new one if no pretrained available ======
        qa.initialize_model(sess, config.train_dir)
        qa.train(sess, [train, dev], config.train_dir)



if __name__ == "__main__":
    run_func()
