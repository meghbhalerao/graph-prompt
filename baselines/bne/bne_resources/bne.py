"""
# Use BNE_SGw pretrained model:
python bne.py --model models/BNE_SGw --fe e:/Emb_SGw.txt --fi names.txt --fo output_BNE_SGw.txt

# Use BNE_SGsc pretrained model:
python bne.py --model models/BNE_SGsc --fe e:/Emb_SGsc.txt --fi names.txt --fo output_BNE_SGsc.txt

# Use average embedding baseline:
python bne.py --model AvgEmb --fe e:/Emb_SGw.txt --fi names.txt --fo output_AvgEmb_SGw.txt

# Export files for embedding projector (https://projector.tensorflow.org/):
python bne.py --model models/BNE_SGw --fe e:/Emb_SGw.txt --fi names.txt --fl labels.txt --fo output_BNE_SGw.txt
python bne.py --model AvgEmb --fe e:/Emb_SGsc.txt --fi names.txt --fl labels.txt --fo output_AvgEmb_SGw.txt

"""

import argparse
import math
import json
from encoder import *
import os
import pickle
import sys

class BNE:
    def __init__(self, session, pretrained_we, params):
        self.session = session
        self.encoder = NameEncoder(params, session)

        # Load pretrained model parameters.
        print('Load pretrained model from {}'.format(arg_model + '/best_model'))

        session.run(tf.global_variables_initializer())
        variables = [_ for _ in tf.global_variables()
                     if _ not in self.encoder.ignored_vars]

        values  = self.session.run(variables)
        names = variables
        weight_dict = {}

        for idx, item in enumerate(names):
            str_item = str(item)
            weight_dict[str_item] = values[idx]
       
        """
        for item in weight_dict:
            shape_str = [x for x in item.split(' ') if 'shape' in str(x)]
            print(shape_str, weight_dict[item].shape)

        """
  
        file_name = "weight_dict.pkl"

        open_file = open(file_name, "wb")
        pickle.dump(weight_dict, open_file)
        open_file.close()

        self.saver = tf.train.Saver(variables, max_to_keep=1)

        self.saver.restore(session, arg_model + '/best_model')

        print("Loaded the model weights into the session ...")

        # Load pretrained word embeddings.
        emb = self.encoder.we_core.eval(session)

        print("Finished getting word embeddings of shape - ", emb.shape)

        offset = 1
        for w in pretrained_we:
            if w in params['w2id']:
                emb[params['w2id'][w] - offset] = pretrained_we[w]
        emb_holder = tf.placeholder(tf.float32, [len(emb), params['we_dim']])
        print("Shape of the embedding placeholder for feed_dict is ", emb_holder.shape)
        init_op = self.encoder.we_core.assign(emb_holder)
        
        session.run(init_op, feed_dict={emb_holder: emb})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str,
                        help='Path to the pretrained encoder.')
    parser.add_argument("--fe", type=str,
                        help='Path to the pretrained word embedding file.')
    parser.add_argument("--fi", type=str,
                        help='Input file which contains one name on each line.')
    parser.add_argument("--fl", type=str,
                        help='(Optional) Input file that stores the labels of '
                             'the input names.')
    parser.add_argument("--fo", type=str,
                        help='Output file. Each line contains an embedding '
                             'associated with an input name.')
    parser.add_argument("--bsize", type=int, default=1024, help='Batch size. Number of names in a batch.')
    print("Entered the \"main\" function of BNE inference ...")

    args = parser.parse_args()
    arg_model = args.model
    arg_fe = args.fe
    arg_fi = args.fi
    arg_fl = args.fl
    arg_fo = args.fo
    arg_bsize = args.bsize
    print("Parsed the arguments passed to the python script ...")
    # Extract word vocabulary from the input names.
    names = []
    with open(arg_fi, encoding='utf8') as f:
        for i, line in enumerate(f):
            if line.strip() != '':
                names.append(line.split('\t')[0].strip())
    print("Parsed the input name file ...")
    name_to_pname = dict([(_, preprocess_name(_)) for _ in names])
    print("The length of the names file is ", len(names))
    ws = []

    for name in name_to_pname.values():
        for w in name.split():
            ws.append(w)
    
    print("The length of the individual words in the composite word is ", len(ws))
    print("Reserve Tokens are ", RESERVE_TKS)
    ws = set([_.lower() for _ in ws]) - set(RESERVE_TKS)
    pretrained_we = load_pretrained_we(arg_fe, ws)
    print("Loaded pretrained We ... ")
    pretrained_ws = set(pretrained_we.keys())

    print('Load {}/{} pretrained word embeddings'
          .format(len(ws & pretrained_ws), len(ws)))
    print('Some words that do not have pretrained embeddings: {}'.format('; '.join(list(ws - pretrained_ws)[:20])))
    ws = (ws & pretrained_ws) - set(RESERVE_TKS)

    w2id = {PAD: 0}
    
    for w in ws:
        w2id[w] = len(w2id)
    for w in RESERVE_TKS:
        if w != PAD:
            w2id[w] = len(w2id)
    print("Created dictionary of word to IDs ...")
    name_vecs = []
    if arg_model == 'AvgEmb':
        # Average embedding baseline.
        for name in names:
            word_vecs = [pretrained_we[_] for _ in name_to_pname[name].split() if _ in pretrained_we]
            if len(word_vecs) == 0:
                v = [0] * 200
            elif len(word_vecs) == 1:
                v = word_vecs[0]
            else:
                v = np.mean(word_vecs, axis=0)
            name_vecs.append(v)

    else:
        print("Doing sequence model word embeddings ... (NOT average word embeddings of individual words)")
        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=config) as session:
            # Load the pretrained model's configuration.

            with open(arg_model + '/params.json', encoding='utf8') as f:
                params = json.load(f)
                params['w2id'] = w2id

            print("Finished loading model/evaluation parameters from the json file here - ", os.path.join(arg_model,'params.json'))

            # Declare and load pretrained model.
            model = BNE(session, pretrained_we, params)
            encoder = model.encoder
            print("Finished loading the encoder from the BNE class ...")
            # Calculate name embeddings.
            pnames = list(set(name_to_pname.values()))
            d_batches = chunk([_.split() for _ in pnames], arg_bsize)

            print("Finished batching the data ..")

            embeddings = []
            print('Calculate name embeddings for {} batches'.format(math.ceil(len(pnames) / arg_bsize)))

            summary_writer = tf.summary.FileWriter(os.path.join("./tb_logs/"), graph = tf.get_default_graph())

            for d_batch in tqdm(iter(d_batches)):
                fd_batch = encoder.get_fd_data(d_batch)
                embeddings += list(session.run(encoder.h, feed_dict=fd_batch))
            pname_to_emb = dict(zip(pnames, embeddings))

        for name in names:
            v = pname_to_emb[name_to_pname[name]]
            name_vecs.append(v)

    # Export the result.
    print('Write result into {}'.format(arg_fo))
    with open(arg_fo, 'w', encoding='utf8') as f:
        for name, vec in zip(names, name_vecs):
            str_vec = ' '.join(['{:.5f}'.format(_) for _ in vec])
            f.write('{}\t{}\n'.format(name, str_vec))

    # Export files for embedding visualization.
    # https://projector.tensorflow.org/
    if arg_fl is not None:
        f_metadata = 'projector/metadata_{}.tsv'.format(arg_fo)
        f_vectors = 'projector/vectors_{}.tsv'.format(arg_fo)
        print('Export embedding projector files into {} and {}'
              .format(f_metadata, f_vectors))
        labels = []
        with open(arg_fl, encoding='utf8') as f:
            for line in f:
                labels.append(line.strip())
        assert len(names) == len(labels) == len(name_vecs)
        with open(f_metadata, 'w', encoding='utf8') as f:
            f.write('Name\tLabel\n')
            for name, label in zip(names, labels):
                f.write('{}\t{}\n'.format(name, label))

        with open(f_vectors, 'w') as f:
            for vec in name_vecs:
                str_vec = '\t'.join(['{:.5f}'.format(_) for _ in vec])
                f.write('{}\n'.format(str_vec))