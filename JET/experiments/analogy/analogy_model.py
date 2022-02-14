'''
Adapted from
https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/embedding/word2vec.py
'''
import numpy as np
import tensorflow as tf
from drgriffis.science.metrics import AP_RR

class Mode:
    ThreeCosAdd = 0
    PairwiseDistance = 1
    ThreeCosMul = 2

    @staticmethod
    def name(x):
        if x == Mode.ThreeCosAdd:
            return 'ThreeCosAdd'
        elif x == Mode.PairwiseDistance:
            return 'PairwiseDistance'
        elif x == Mode.ThreeCosMul:
            return 'ThreeCosMul'
        else:
            return str(x)

class AnalogyModel:
    '''
    MAP/MRR not reported if not using multi_answer property
    '''

    def __init__(self, session, embed_array, mode=Mode.ThreeCosAdd):
        self._session = session
        self._mode = mode
        self._vocab_size = embed_array.shape[0]
        self._dim = embed_array.shape[1]
        self._build()
        self._session.run(self._embed_var.assign(self._embed_ph), feed_dict={self._embed_ph: embed_array})

    def eval(self, analogies, analogy_embeds, batch_size=500, report_top_k=5, log=None):
        analogies = np.array(analogies, dtype=np.int32)
        analogy_embeds = np.array(analogy_embeds, dtype=np.float32)
        batch_start, total = 0, analogies.shape[0]

        correct = 0  # accuracy evaluation
        average_precision_sum, reciprocal_rank_sum = 0, 0  # MAP/MRR evaluation
        skipped = 0

        if log: log.track(message='  >> Predictions: {1}/%d' % total)

        predictions = []
        while batch_start < total:
            limit = batch_start + batch_size
            sub_ixes = analogies[batch_start:limit, :]
            sub_embs = analogy_embeds[batch_start:limit, :, :]
            _, ix = self._predict(sub_embs)
            batch_start = limit

            for question in range(len(sub_ixes)):
                
                # get the set of correct answers
                expected = set(sub_ixes[question, 3:])
                expected.discard(-2)
                expected.discard(-1)
                if len(expected) == 0:
                    skipped += 1
                    predictions.append((False, -1, [-1]))
                    continue
                
                # accuracy evaluation
                is_correct = False
                for j in range(4):
                    # make sure that this isn't one of the other terms in the analogy
                    # (if so, it doesn't count)
                    if ix[question, j] in sub_ixes[question, :3]:
                        continue
                    # check if this is (one of) the right answer(s)
                    elif ix[question, j] in expected:
                        correct += 1
                        is_correct = True
                        break
                    # otherwise, we got it wrong
                    else:
                        break

                # MAP/MRR evaluation
                #print('Question -- %d : %d :: %d : %s' % (sub_ixes[question,0], sub_ixes[question,1], sub_ixes[question,2], expected))
                (ap, rr) = AP_RR(expected, ix[question])

                # reduce to average precision
                average_precision_sum += ap
                # add rr to reciprocal rank summer
                reciprocal_rank_sum += rr

                predictions.append((is_correct, len(ix[question]), ix[question,:report_top_k]))

            if log: log.tick(min(batch_start, total))

        # calculate MAP and MRR
        if (total-skipped) > 0:
            mean_average_precision = (average_precision_sum / (total-skipped))
            mean_reciprocal_rank = (reciprocal_rank_sum / (total-skipped))
        else:
            mean_average_precision = 0
            mean_reciprocal_rank = 0

        return correct, mean_average_precision, mean_reciprocal_rank, total, skipped, predictions

    def _build(self):
        with tf.device('/cpu:0'):
            self._embed_ph = tf.placeholder(tf.float32, [self._vocab_size, self._dim])
            self._embed_var = embed_var = tf.Variable(tf.constant(0.0, shape=[self._vocab_size, self._dim]), trainable=False)

            analogy_a = tf.placeholder(dtype=tf.float32, shape=[None, self._dim])
            analogy_b = tf.placeholder(dtype=tf.float32, shape=[None, self._dim])
            analogy_c = tf.placeholder(dtype=tf.float32, shape=[None, self._dim])

            nemb = tf.nn.l2_normalize(embed_var, 1)

            if self._mode == Mode.ThreeCosAdd:
                target = (analogy_b - analogy_a) + analogy_c

                dist = tf.matmul(target, nemb, transpose_b=True)
            elif self._mode == Mode.PairwiseDistance:
                example_offset = analogy_b - analogy_a
                query_offsets = embed_var - analogy_c

                example_norm = tf.sqrt(tf.reduce_sum(example_offset**2))
                query_norms = tf.sqrt(tf.reduce_sum(query_offsets**2, reduction_indices=[1]))

                dist = (
                    tf.matmul(example_offset, query_offsets, transpose_b=True) /
                    tf.transpose(example_norm * query_norms)
                )
            elif self._mode == Mode.ThreeCosMul:
                numerator_left = tf.matmul(analogy_b, nemb, transpose_b=True) / tf.sqrt(tf.reduce_sum(analogy_b**2))
                numerator_right = tf.matmul(analogy_c, nemb, transpose_b=True) / tf.sqrt(tf.reduce_sum(analogy_c**2))
                denominator = tf.matmul(analogy_a, nemb, transpose_b=True) / tf.sqrt(tf.reduce_sum(analogy_a**2))

                dist = (numerator_left * numerator_right) / (denominator + 0.000001)

            nearest_dists, pred_ix = tf.nn.top_k(dist, self._vocab_size)

            self._analogy_a = analogy_a
            self._analogy_b = analogy_b
            self._analogy_c = analogy_c
            self._analogy_pred_ix = pred_ix
            self._analogy_pred_dists = nearest_dists

    def _predict(self, analogy_embs):
        dists, idx = self._session.run([self._analogy_pred_dists, self._analogy_pred_ix], {
            self._analogy_a : analogy_embs[:,0,:],
            self._analogy_b : analogy_embs[:,1,:],
            self._analogy_c : analogy_embs[:,2,:],
        })
        return dists, idx
