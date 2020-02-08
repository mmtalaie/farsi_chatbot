import tkinter
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

from collections import Counter
import os
import nltk


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
        msgLowerCase = [w.lower() for w in nltk.word_tokenize(humanMsg)]
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


def Enter_pressed(event):
    input_get = input_field.get()
    print(input_get)
    bot_reply = generateReply(input_get)
    if (len(input_get.strip()) > 0):
        messages.insert(INSERT, '\nYou says: \t%s' % input_get)
    if (len(bot_reply.strip()) > 0):
        messages.insert(INSERT, '\nBot says: \t%s' % bot_reply)
    input_user.set('')
    messages.see(tkinter.END)
    return "break"


window = ThemedTk()
window.set_theme("blue")

# window = Tk()
window.geometry('300x450')
window.title("Digital Imprint of You!")

messages = Text(window)
messages.insert(INSERT, '')
messages.pack()

input_user = StringVar()
input_field = ttk.Entry(window, text=input_user)
input_field.pack(side=BOTTOM, fill=X)

# frame = Frame(window)
input_field.bind("<Return>", Enter_pressed)
input_field.pack()


btn = Button(window, text='Send', command=Enter_pressed(''))
btn.bind('<Button-1>', Enter_pressed)
btn.pack(side=RIGHT, fill=X)


window.mainloop()
