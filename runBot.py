
import tensorflow as tf
from hazm import word_tokenize
import numpy as np
from collections import Counter
import os

from sklearn.model_selection import train_test_split


alphas = 'ابپتسجچهخدذرزژسشصضطظعغفقکگلمنوهییٔءآاًهٔة'

MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
DATA_DIR_PATH = 'data'
MAX_VOCAB_SIZE = 30000

marker_start = '<begin>'
marker_end = '<end>'
marker_unknown = '<unk>'
marker_pad = '<pad>'

# defines permissible characters for the chatbot


def permissible_chars(word):

    for char in word:
        if char in alphas:
            return True

    return False


# standard step - reset computation graphs
tf.reset_default_graph()

# 2 more for start and stop markers
input_seq_len = 15
output_seq_len = input_seq_len+2

# To parse the input yml files and create word2index and index2word mappings

target_counter = Counter()
input_counter = Counter()

input_texts = []
target_texts = []

# Parser base code of GuntherCox dataset obtained from link below and modified as per requirement.
# https://github.com/kushagra2101/ChatCrazie/blob/master/train_seq2seq.py

for file in os.listdir(DATA_DIR_PATH):
    filepath = os.path.join(DATA_DIR_PATH, file)
    if os.path.isfile(filepath):
        print('processing file: ', file)
        lines = open(filepath, 'rt', encoding='utf8').read().split('\n')
        prev_words = []
        for line in lines:

            if line.startswith('- - '):
                prev_words = []

            if line.startswith('- - ') or line.startswith('  - '):
                line = line.replace('- - ', '')
                line = line.replace('  - ', '')
                next_words = [w.lower() for w in word_tokenize(line)]
                next_words = [w for w in next_words if permissible_chars(w)]
                if len(next_words) > MAX_TARGET_SEQ_LENGTH:
                    next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

                if len(prev_words) > 0:
                    input_texts.append(prev_words)
                    for w in prev_words:
                        input_counter[w] += 1

                    target_words = next_words[:]
                    for w in target_words:
                        target_counter[w] += 1
                    target_texts.append(target_words)

                prev_words = next_words


for idx, (input_words, target_words) in enumerate(zip(input_texts, target_texts)):
    if idx < 20:
        print([input_words, target_words])

input_w2i, input_i2w, target_w2i, target_i2w = {}, {}, {}, {}

### Creating Word2index and Index2word, forward and reverse mapping ###
# we will create dictionaries to provide a unique integer for each word.
input_w2i[marker_unknown] = 0
input_w2i[marker_pad] = 1
# filter out the rare words
for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
    input_w2i[word[0]] = idx+2

# inverse dictionary for vocab_to_int.
input_i2w = dict([(idx, word) for word, idx in input_w2i.items()])

# we will create dictionaries to provide a unique integer for each word.
target_w2i[marker_unknown] = 0
target_w2i[marker_pad] = 1
target_w2i[marker_start] = 2
target_w2i[marker_end] = 3
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_w2i[word[0]] = idx+4

# inverse dictionary for vocab_to_int.
target_i2w = dict([(idx, word) for word, idx in target_w2i.items()])


###########################################

# inputVocabLen = len(input_word2idx)
# targetVocabLen = len(target_word2idx)

###########################################

# if the word is not found then default with 0.
# 0 in index means the word is unknown (<unk>)
x = [[input_w2i.get(word, 0) for word in sentence] for sentence in input_texts]
y = [[target_w2i.get(word, 0) for word in sentence]
     for sentence in target_texts]

inputVocabLen = len(input_w2i)
targetVocabLen = len(target_w2i)

# Compute softmax values for each sets of scores in x.
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# decode output sequence
def decode_output(output_seq):
    words = []
    for i in range(output_seq_len):
        smax = softmax(output_seq[i])
        idx = np.argmax(smax)
        words.append(target_i2w[idx])
    return words

# To predict response (inference) use the same model as defined above with forward feed


def generateReply(humanMsg):

    if (len(humanMsg) == 0):
        return ''

    with tf.Graph().as_default():

        replyMsg = ""

        # same format as in model building
        encoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None],
                                         name='encoder{}'.format(i)) for i in range(input_seq_len)]
        decoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None],
                                         name='decoder{}'.format(i)) for i in range(output_seq_len)]

        # output projection
        size = 512
        w_t = tf.get_variable('proj_w', [targetVocabLen, size], tf.float32)
        b = tf.get_variable('proj_b', [targetVocabLen], tf.float32)
        w = tf.transpose(w_t)
        output_projection = (w, b)

        # feed_previous is set to true so that output at time t can be fed as input at time t+1
        outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            encoder_inputs,
            decoder_inputs,
            tf.contrib.rnn.BasicLSTMCell(size),
            num_encoder_symbols=inputVocabLen,
            num_decoder_symbols=targetVocabLen,
            embedding_size=100,
            feed_previous=True,
            output_projection=output_projection,
            dtype=tf.float32)
        # ops for projecting outputs
        outputs_proj = [tf.matmul(outputs[i],
                                  output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

        # Clean and Format incoming msg by humans.
        # It is better to do the same clean/format as the data preprocessing steps
        # for the algorithm to predict next words more accurately
        msgLowerCase = [w.lower() for w in word_tokenize(humanMsg)]
        msg = [w for w in msgLowerCase if permissible_chars(w)]
        if len(msg) > input_seq_len:
            msg = msg[0:input_seq_len-1]

        human_msg_encoded = [input_w2i.get(word, 0) for word in msg]

        # Fill in with padding marker
        for k in range(input_seq_len - len(human_msg_encoded)):
            human_msg_encoded = human_msg_encoded + [input_w2i[marker_pad]]

        # restore all variables - use the last checkpoint saved
        saver = tf.train.Saver()
        path = tf.train.latest_checkpoint('checkpoints')

        with tf.Session() as sess:
            # restore
            saver.restore(sess, path)

            # feed data into placeholders
            feed = {}
            for i in range(input_seq_len):
                feed[encoder_inputs[i].name] = np.array(
                    [human_msg_encoded[i]], dtype=np.int32)

            feed[decoder_inputs[0].name] = np.array(
                [target_w2i[marker_start]], dtype=np.int32)

            # translate
            output_sequences = sess.run(outputs_proj, feed_dict=feed)

            ouput_seq = [output_sequences[j][0] for j in range(output_seq_len)]
            # decode output sequence
            words = decode_output(ouput_seq)

            for i in range(len(words)):
                if words[i] not in [marker_end, marker_pad, marker_start]:
                    replyMsg += words[i] + ' '

        print(replyMsg)
        return replyMsg


import tkinter as tk



def Enter_pressed(event):
    input_get = input_field.get()
    print(input_get)
    bot_reply = generateReply(input_get)
    if (len(input_get.strip()) > 0):
        messages.insert(tk.INSERT, '\nYou says: \t%s' % input_get)
    if (len(bot_reply.strip()) > 0):
        messages.insert(tk.INSERT, '\nBot says: \t%s' % bot_reply)
    input_user.set('')
    messages.see(tk.END)
    return "break"

from ttkthemes import ThemedTk
window = ThemedTk()
window.set_theme("blue")

# window = Tk()
window.geometry('300x450')
window.title("Digital Imprint of You!")

messages =  tk.Text(window)
messages.insert(tk.INSERT, '')
messages.pack()

input_user = tk.StringVar()
input_field = tk.Entry(window, text=input_user)
# input_field.pack(side=tk.BOTTOM, fill=X)
input_field.pack()

# frame = Frame(window)
input_field.bind("<Return>", Enter_pressed)
input_field.pack()


btn = tk.Button(window, text='Send', command=Enter_pressed(''))
btn.bind('<Button-1>', Enter_pressed)
# btn.pack(side=tk.RIGHT, fill=X)
btn.pack()

window.mainloop()
