import tensorflow as tf
import numpy as np
import math
import sys
from flask import Flask,request,jsonify



from config import FLAGS
from model import Seq2Seq
from dialog import Dialog


app = Flask(__name__)

class ChatBot:

    def __init__(self, voc_path, train_dir):
        self.dialog = Dialog()
        self.dialog.load_vocab(voc_path)

        self.model = Seq2Seq(self.dialog.vocab_size)

        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(train_dir)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def run(self):
        sys.stdout.write("> ")
        sys.stdout.flush()
        line = sys.stdin.readline()

        while line:
            print (self.get_replay(line.strip()))

            sys.stdout.write("\n> ")
            sys.stdout.flush()

            line = sys.stdin.readline()

    def decode(self, enc_input, dec_input):
        if type(dec_input) is np.ndarray:
            dec_input = dec_input.tolist()

        input_len = int(math.ceil((len(enc_input) + 1) * 1.5))

        enc_input, dec_input, _ = self.dialog.transform(enc_input, dec_input,
                                                        input_len,
                                                        FLAGS.max_decode_len)

        return self.model.predict(self.sess, [enc_input], [dec_input])

    def get_replay(self, msg):
        enc_input = self.dialog.tokenizer(msg)
        enc_input = self.dialog.tokens_to_ids(enc_input)
        dec_input = []


        curr_seq = 0
        for i in range(FLAGS.max_decode_len):
            outputs = self.decode(enc_input, dec_input)
            if self.dialog.is_eos(outputs[0][curr_seq]):
                break
            elif self.dialog.is_defined(outputs[0][curr_seq]) is not True:
                dec_input.append(outputs[0][curr_seq])
                curr_seq += 1

        reply = self.dialog.decode([dec_input], True)

        return reply
@app.route('/keyboard')

def Keyboard():
    keyList= ["temperature","sensor"]
    keyboardList = {'type': 'buttons', 'buttons': keyList}
    return jsonify(keyboardList)

@app.route('/message', methods=['POST'])

def GetMessage():
    content = request.get_json()
    print content
    text = content['content']
    textContent = {"text":text}
    textMessage = {"message":textContent}
    return jsonify(textMessage)

def main(_):
    print ("waiting...\n")

    chatbot = ChatBot(FLAGS.voc_path, FLAGS.train_dir)
    chatbot.run()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    tf.app.run()