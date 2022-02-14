'''
Log-linear model for entity linking.

Implementation of the third scoring model (f^{c,p}) described in
Sabbir et al, "Knowledge-based biomedical word sense disambiguation
with neural concept embeddings." IEEE Symposium on Bioinformatics
and Bioengineering, 2017.
'''

import numpy as np
import tensorflow as tf
from .params import EntityComboMethods

MIN_FLOAT = np.finfo(dtype=np.float32).min

class LinearSabbirLinkerC:
    
    def __init__(self, session, ctx_embed_arr, entity_embed_arr, params, trainable_embeds=False, debug=False, secondary_entity_embed_arr=None):
        self._session = session
        self._p = params
        self._build(trainable_embeds=trainable_embeds, debug=debug)
        self._session.run(
            [
                self._ctx_embed_matrix.assign(self._ctx_embed_ph),
                self._entity_embed_matrix.assign(self._entity_embed_ph)
            ],
            feed_dict={
                self._ctx_embed_ph : ctx_embed_arr,
                self._entity_embed_ph : entity_embed_arr
            }
        )

        if self._p._secondary_entity_vocab_size > 0:
            self._session.run(
                [self._secondary_entity_embed_matrix.assign(self._secondary_entity_embed_ph)],
                feed_dict={
                    self._secondary_entity_embed_ph : secondary_entity_embed_arr
                }
            )

    def getPredictions(self, batch_ctx_window_ixes, batch_ctx_window_masks, batch_entity_ixes, batch_entity_binary_masks,
            batch_secondary_entity_ixes=None, batch_mention_ixes=None, batch_mention_masks=None, oracle=False):
        batch_ctx_window_ixes = np.array(batch_ctx_window_ixes)
        batch_ctx_window_masks = np.array(batch_ctx_window_masks)
        batch_entity_ixes = np.array(batch_entity_ixes)
        batch_entity_binary_masks = np.array(batch_entity_binary_masks)
        if (not batch_secondary_entity_ixes is None):
            batch_secondary_entity_ixes = np.array(batch_secondary_entity_ixes)
        if (not batch_mention_ixes is None):
            batch_mention_ixes = np.array(batch_mention_ixes)
        if (not batch_mention_masks is None):
            batch_mention_masks = np.array(batch_mention_masks)

        batch_entity_value_masks = np.ones(shape=batch_entity_binary_masks.shape) * MIN_FLOAT

        # validate inputs
        if len(batch_ctx_window_ixes.shape) != 2:
            print("[ERROR] Must provide batch context window ixes as [ [window 1], [window 2], ... ]")
            print(batch_ctx_window_ixes)
            return
        if len(batch_ctx_window_masks.shape) != 2:
            print("[ERROR] Must provide batch context window masks as [ [window 1], [window 2], ... ]")
            print(batch_ctx_window_masks)
            return
        if len(batch_entity_ixes.shape) != 2:
            print("[ERROR] Must provide batch entity ixes as [ [set 1], [set 2], ... ]")
            print(batch_entity_ixes)
            return
        if len(batch_entity_binary_masks.shape) != 2:
            print("[ERROR] Must provide batch entity masks as [ [set 1], [set 2], ... ]")
            print(batch_entity_binary_masks)
            return
        if (not batch_secondary_entity_ixes is None) and len(batch_secondary_entity_ixes.shape) != 2:
            print("[ERROR] Must provide batch secondary entity ixes as [ [set 1], [set 2], ... ]")
            print(batch_secondary_entity_ixes)
            return
        if (not batch_mention_ixes is None) and len(batch_mention_ixes.shape) != 2:
            print("[ERROR] Must provide batch mention ixes as [ [mention 1], [mention 2], ... ]")
            print(batch_mention_ixes)
            return
        if (not batch_mention_masks is None) and len(batch_mention_masks.shape) != 2:
            print("[ERROR] Must provide batch mention masks as [ [mention 1], [mention 2], ... ]")
            print(batch_mention_masks)
            return


        # and use them to get context window probabilities
        feed_dict = {
            self._ctx_ixes : batch_ctx_window_ixes,
            self._ctx_mask : batch_ctx_window_masks,
            self._entity_ixes : batch_entity_ixes,
            self._entity_binary_mask : batch_entity_binary_masks,
            self._entity_value_mask : batch_entity_value_masks
        }
        if (not batch_secondary_entity_ixes is None):
            feed_dict[self._secondary_entity_ixes] = batch_secondary_entity_ixes
        if (not batch_mention_ixes is None):
            feed_dict[self._mention_ixes] = batch_mention_ixes
        if (not batch_mention_masks is None):
            feed_dict[self._mention_mask] = batch_mention_masks

        execs = [self._entity_predictions, self._final_entity_scores]
        if oracle:
            execs.append(self._entity_only_predictions)
            if (not batch_secondary_entity_ixes is None):
                execs.append(self._secondary_entity_only_predictions)
        results = self._session.run(
            execs,
            feed_dict = feed_dict
        )

        return results

    def _build(self, trainable_embeds=False, debug=False):
        if debug:
            print(" -- DEBUG MESSAGES ENABLED --")

        # inputs
        self._ctx_ixes = tf.placeholder(dtype=tf.int32, shape=[None, self._p._window_size*2])
        self._entity_ixes = tf.placeholder(dtype=tf.int32, shape=[None, self._p._max_num_entities])
        self._secondary_entity_ixes = tf.placeholder(dtype=tf.int32, shape=[None, self._p._max_num_entities])
        self._mention_ixes = tf.placeholder(dtype=tf.int32, shape=[None, self._p._max_mention_size])

        # masks
        self._ctx_mask = tf.placeholder(dtype=tf.float32, shape=[None, self._p._window_size*2])
        self._entity_binary_mask = tf.placeholder(dtype=tf.int32, shape=[None, self._p._max_num_entities])
        self._entity_value_mask = tf.placeholder(dtype=tf.float32, shape=[None, self._p._max_num_entities])
        self._mention_mask = tf.placeholder(dtype=tf.float32, shape=[None, self._p._max_mention_size])

        if debug:
            ctx_ixes = tf.Print(self._ctx_ixes, [self._ctx_ixes], summarize=100, message="Ctx Ixes")
            entity_ixes = tf.Print(self._entity_ixes, [self._entity_ixes], summarize=100, message="Entity Ixes")
            secondary_entity_ixes = tf.Print(self._secondary_entity_ixes, [self._secondary_entity_ixes], summarize=100, message="Secondary Entity Ixes")
            ctx_mask = tf.Print(self._ctx_mask, [self._ctx_mask], summarize=100, message="Ctx Mask")
            mention_ixes = tf.Print(self._ctx_ixes, [self._ctx_ixes], summarize=100, message="Mention Ixes")
            mention_mask = tf.Print(self._mention_mask, [self._mention_mask], summarize=100, message="Mention Mask")
        else:
            ctx_ixes = self._ctx_ixes
            entity_ixes = self._entity_ixes
            secondary_entity_ixes = self._secondary_entity_ixes
            ctx_mask = self._ctx_mask
            mention_ixes = self._mention_ixes
            mention_mask = self._mention_mask

        # embedding matrices
        self._ctx_embed_ph = tf.placeholder(tf.float32, [self._p._ctx_vocab_size, self._p._ctx_dim])
        self._ctx_embed_matrix = tf.Variable(
            tf.constant(0.0, shape=[self._p._ctx_vocab_size, self._p._ctx_dim]),
            trainable=trainable_embeds
        )
        self._entity_embed_ph = tf.placeholder(tf.float32, [self._p._entity_vocab_size, self._p._entity_dim])
        self._entity_embed_matrix = tf.Variable(
            tf.constant(0.0, shape=[self._p._entity_vocab_size, self._p._entity_dim]),
            trainable=trainable_embeds
        )
        self._secondary_entity_embed_ph = tf.placeholder(tf.float32, [self._p._secondary_entity_vocab_size, self._p._secondary_entity_dim])
        self._secondary_entity_embed_matrix = tf.Variable(
            tf.constant(0.0, shape=[self._p._secondary_entity_vocab_size, self._p._secondary_entity_dim]),
            trainable=trainable_embeds
        )

        if debug:
            ctx_embed_matrix = tf.Print(self._ctx_embed_matrix, [self._ctx_embed_matrix], summarize=100, message="Ctx Embed Matr")
            entity_embed_matrix = tf.Print(self._entity_embed_matrix, [self._entity_embed_matrix], summarize=100, message="Entity Embed Matr")
            secondary_entity_embed_matrix = tf.Print(self._secondary_entity_embed_matrix, [self._secondary_entity_embed_matrix], summarize=100, message="Secondary Entity Embed Matr")
        else:
            ctx_embed_matrix = self._ctx_embed_matrix
            entity_embed_matrix = self._entity_embed_matrix
            secondary_entity_embed_matrix = self._secondary_entity_embed_matrix

        # grab the embeddings of only the window terms
        # yields < BS x WS x ES >
        ctx_window_embeds = tf.gather(
            ctx_embed_matrix,
            ctx_ixes
        )
        mention_embeds = tf.gather(
            ctx_embed_matrix,
            mention_ixes
        )

        if debug:
            ctx_window_embeds = tf.Print(ctx_window_embeds, [ctx_window_embeds], summarize=100, message="Ctx Window Embeds")
        
        # apply the mask
        ctx_mask = tf.reshape(
            ctx_mask,
            [-1, self._p._window_size*2, 1]
        )
        mention_mask = tf.reshape(
            mention_mask,
            [-1, self._p._max_mention_size, 1]
        )
        if debug:
            ctx_mask = tf.Print(ctx_mask, [ctx_mask], summarize=100, message="Reshaped ctx mask")
            mention_mask = tf.Print(mention_mask, [mention_mask], summarize=100, message="Reshaped mention mask")

        ctx_mask = tf.tile(
            ctx_mask,
            [1, 1, self._p._ctx_dim]
        )
        mention_mask = tf.tile(
            mention_mask,
            [1, 1, self._p._ctx_dim]
        )
        if debug:
            ctx_mask = tf.Print(ctx_mask, [ctx_mask], summarize=100, message="Tiled reshaped ctx mask")
            mention_mask = tf.Print(mention_mask, [mention_mask], summarize=100, message="Tiled reshaped mention mask")

        ctx_window_embeds = ctx_window_embeds * ctx_mask
        mention_embeds = mention_embeds * mention_mask

        if debug:
            ctx_window_embeds = tf.Print(ctx_window_embeds, [ctx_window_embeds], summarize=100, message="Masked ctx window embeds")
            mention_embeds = tf.Print(mention_embeds, [mention_embeds], summarize=100, message="Masked mention embeds")

        # mean over the window
        # yields < BS x ES >
        ctx_window_embeds = tf.reduce_sum(
            ctx_window_embeds,
            axis=1
        )
        mention_embeds = tf.reduce_sum(
            mention_embeds,
            axis=1
        )
        if debug:
            ctx_window_embeds = tf.Print(ctx_window_embeds, [ctx_window_embeds], summarize=100, message="Summed window embeds")
            mention_embeds = tf.Print(mention_embeds, [mention_embeds], summarize=100, message="Summed mention embeds")

        ctx_normalizer = tf.reduce_sum(
            ctx_mask,
            axis=1
        )
        mention_normalizer = tf.reduce_sum(
            mention_mask,
            axis=1
        )
        if debug:
            ctx_normalizer = tf.Print(ctx_normalizer, [ctx_normalizer], summarize=100, message="Ctx normalizer")
            mention_normalizer = tf.Print(mention_normalizer, [mention_normalizer], summarize=100, message="Mention normalizer")

        ctx_window_embeds = ctx_window_embeds / ctx_normalizer
        mention_embeds = mention_embeds / mention_normalizer
        if debug:
            ctx_window_embeds = tf.Print(ctx_window_embeds, [ctx_window_embeds], summarize=100, message="Mean ctx window embeds")
            mention_embeds = tf.Print(mention_embeds, [mention_embeds], summarize=100, message="Mean mention embeds")

        # grab the entities to use
        # yields < BS x NE x ES >
        entity_set_embeds = tf.gather(
            entity_embed_matrix,
            entity_ixes,
        )
        secondary_entity_set_embeds = tf.gather(
            secondary_entity_embed_matrix,
            secondary_entity_ixes
        )

        if debug:
            entity_set_embeds = tf.Print(entity_set_embeds, [entity_set_embeds], summarize=100, message="Entity Set Embeds")
            secondary_entity_set_embeds = tf.Print(secondary_entity_set_embeds, [secondary_entity_set_embeds], summarize=100, message="Secondary Entity Set Embeds")

        # norm their embeddings
        normed_entity_set_embeds = tf.nn.l2_normalize(
            entity_set_embeds,
            dim=2
        )
        normed_secondary_entity_set_embeds = tf.nn.l2_normalize(
            secondary_entity_set_embeds,
            dim=2
        )
        normed_ctx_window_embeds = tf.nn.l2_normalize(
            ctx_window_embeds,
            dim=1
        )
        normed_mention_embeds = tf.nn.l2_normalize(
            mention_embeds,
            dim=1
        )

        if debug:
            normed_entity_set_embeds = tf.Print(normed_entity_set_embeds, [normed_entity_set_embeds], summarize=100, message="Normed Entity Set Embeds")
            normed_secondary_entity_set_embeds = tf.Print(normed_secondary_entity_set_embeds, [normed_secondary_entity_set_embeds], summarize=100, message="Normed Secondary Entity Set Embeds")
            normed_ctx_window_embeds = tf.Print(normed_ctx_window_embeds, [normed_ctx_window_embeds], summarize=100, message="Normed ctx window embeds")
            normed_mention_embeds = tf.Print(normed_mention_embeds, [normed_mention_embeds], summarize=100, message="Normed mention embeds")

        normed_ctx_window_embeds = tf.reshape(normed_ctx_window_embeds, [-1,1, self._p._ctx_dim])
        normed_mention_embeds = tf.reshape(normed_mention_embeds, [-1,1, self._p._ctx_dim])
        if debug:
            normed_ctx_window_embeds = tf.Print(normed_ctx_window_embeds, [normed_ctx_window_embeds], summarize=100, message="Reshaped normed ctx window embeds")
            normed_mention_embeds = tf.Print(normed_mention_embeds, [normed_mention_embeds], summarize=100, message="Reshaped normed mention embeds")

        # dot products with the mean context window
        # yields < BS x NE >
        entity_ctx_cos_sim = tf.matmul(
            normed_entity_set_embeds,
            normed_ctx_window_embeds,
            transpose_b=True
        )
        entity_mention_cos_sim = tf.matmul(
            normed_entity_set_embeds,
            normed_mention_embeds,
            transpose_b=True
        )
        if self._p._secondary_entity_vocab_size > 0:
            secondary_entity_ctx_cos_sim = tf.matmul(
                normed_secondary_entity_set_embeds,
                normed_ctx_window_embeds,
                transpose_b=True
            )
            secondary_entity_mention_cos_sim = tf.matmul(
                normed_secondary_entity_set_embeds,
                normed_mention_embeds,
                transpose_b=True
            )

        if debug:
            entity_ctx_cos_sim = tf.Print(entity_ctx_cos_sim, [entity_ctx_cos_sim], summarize=100, message="Entity ctx cos sim")
            entity_mention_cos_sim = tf.Print(entity_mention_cos_sim, [entity_mention_cos_sim], summarize=100, message="Entity mention cos sim")
            if self._p._secondary_entity_vocab_size > 0:
                secondary_entity_ctx_cos_sim = tf.Print(secondary_entity_ctx_cos_sim, [secondary_entity_ctx_cos_sim], summarize=100, message="Secondary Entity ctx cos sim")
                secondary_entity_mention_cos_sim = tf.Print(secondary_entity_mention_cos_sim, [secondary_entity_mention_cos_sim], summarize=100, message="Secondary Entity mention cos sim")

        entity_ctx_cos_sim = tf.reshape(
            entity_ctx_cos_sim,
            [-1, self._p._max_num_entities]
        )
        entity_mention_cos_sim = tf.reshape(
            entity_mention_cos_sim,
            [-1, self._p._max_num_entities]
        )
        if self._p._secondary_entity_vocab_size > 0:
            secondary_entity_ctx_cos_sim = tf.reshape(
                secondary_entity_ctx_cos_sim,
                [-1, self._p._max_num_entities]
            )
            secondary_entity_mention_cos_sim = tf.reshape(
                secondary_entity_mention_cos_sim,
                [-1, self._p._max_num_entities]
            )
        if debug:
            entity_ctx_cos_sim = tf.Print(entity_ctx_cos_sim, [entity_ctx_cos_sim], summarize=100, message="Reshaped entity ctx cos sim")
            entity_mention_cos_sim = tf.Print(entity_mention_cos_sim, [entity_mention_cos_sim], summarize=100, message="Reshaped entity mention cos sim")
            if self._p._secondary_entity_vocab_size > 0:
                secondary_entity_ctx_cos_sim = tf.Print(secondary_entity_ctx_cos_sim, [secondary_entity_ctx_cos_sim], summarize=100, message="Reshaped secondary entity ctx cos sim")
                secondary_entity_mention_cos_sim = tf.Print(secondary_entity_mention_cos_sim, [secondary_entity_mention_cos_sim], summarize=100, message="Reshaped secondary entity mention cos sim")

        # calculate the projection of the context onto the entity
        ctx_projection = tf.reshape(
            tf.abs(
                tf.matmul(
                    normed_entity_set_embeds,
                    tf.reshape(ctx_window_embeds, [-1,1,self._p._ctx_dim]),
                    transpose_b=True
                )
            ),
            [-1, self._p._max_num_entities]
        )
        mention_projection = tf.reshape(
            tf.abs(
                tf.matmul(
                    normed_entity_set_embeds,
                    tf.reshape(mention_embeds, [-1,1,self._p._ctx_dim]),
                    transpose_b=True
                )
            ),
            [-1, self._p._max_num_entities]
        )
        if self._p._secondary_entity_vocab_size > 0:
            ctx_secondary_projection = tf.reshape(
                tf.abs(
                    tf.matmul(
                        normed_secondary_entity_set_embeds,
                        tf.reshape(ctx_window_embeds, [-1,1,self._p._ctx_dim]),
                        transpose_b=True
                    )
                ),
                [-1, self._p._max_num_entities]
            )
            mention_secondary_projection = tf.reshape(
                tf.abs(
                    tf.matmul(
                        normed_secondary_entity_set_embeds,
                        tf.reshape(mention_embeds, [-1,1,self._p._ctx_dim]),
                        transpose_b=True
                    )
                ),
                [-1, self._p._max_num_entities]
            )
        if debug:
            ctx_projection = tf.Print(ctx_projection, [ctx_projection], summarize=100, message="Projection of ctx onto entity")
            mention_projection = tf.Print(mention_projection, [mention_projection], summarize=100, message="Projection of mention onto entity")
            if self._p._secondary_entity_vocab_size > 0:
                ctx_secondary_projection = tf.Print(ctx_secondary_projection, [ctx_secondary_projection], summarize=100, message="Projection of ctx onto secondary entity")
                mention_secondary_projection = tf.Print(mention_secondary_projection, [mention_secondary_projection], summarize=100, message="Projection of mention onto secondary entity")

        # multiply the two to get the entity score
        entity_ctx_scores = ctx_projection * entity_ctx_cos_sim
        entity_mention_scores = mention_projection * entity_mention_cos_sim
        if self._p._secondary_entity_vocab_size > 0:
            secondary_entity_ctx_scores = ctx_secondary_projection * secondary_entity_ctx_cos_sim
            secondary_entity_mention_scores = mention_secondary_projection * secondary_entity_mention_cos_sim

        if debug:
            entity_ctx_scores = tf.Print(entity_ctx_scores, [entity_ctx_scores], summarize=100, message="Entity ctx scores")
            entity_mention_scores = tf.Print(entity_mention_scores, [entity_mention_scores], summarize=100, message="Entity mention scores")
            if self._p._secondary_entity_vocab_size > 0:
                secondary_entity_ctx_scores = tf.Print(secondary_entity_ctx_scores, [secondary_entity_ctx_scores], summarize=100, message="Secondary entity ctx scores")
                secondary_entity_mention_scores = tf.Print(secondary_entity_mention_scores, [secondary_entity_mention_scores], summarize=100, message="Secondary entity mention scores")

        # combine ctx/mention scores as appropriate
        if self._p._using_mention:
            entity_scores = entity_ctx_scores * entity_mention_scores
            if self._p._secondary_entity_vocab_size > 0:
                secondary_entity_scores = secondary_entity_ctx_scores * secondary_entity_mention_scores
        else:
            entity_scores = entity_ctx_scores
            if self._p._secondary_entity_vocab_size > 0:
                secondary_entity_scores = secondary_entity_ctx_scores

        self._entity_scores = entity_scores
        if self._p._secondary_entity_vocab_size > 0:
            self._secondary_entity_scores = secondary_entity_scores

        # apply the entity mask to ignore missing/too many entities
        self._entity_scores = tf.where(
            tf.equal(self._entity_binary_mask, 1),
            x=self._entity_scores,
            y=self._entity_value_mask
        )
        if self._p._secondary_entity_vocab_size > 0:
            self._secondary_entity_scores = tf.where(
                tf.equal(self._entity_binary_mask, 1),
                x=self._secondary_entity_scores,
                y=self._entity_value_mask
            )

        if debug:
            self._entity_scores = tf.Print(self._entity_scores, [self._entity_scores], summarize=100, message="Masked entity scores")
            if self._p._secondary_entity_vocab_size > 0:
                self._secondary_entity_scores = tf.Print(self._secondary_entity_scores, [self._secondary_entity_scores], summarize=100, message="Masked secondary entity scores")

        # combine scores, if using secondary entities
        if self._p._secondary_entity_vocab_size > 0:
            if self._p._entity_combination == EntityComboMethods.Sum:
                self._final_entity_scores = self._entity_scores + self._secondary_entity_scores
            elif self._p._entity_combination == EntityComboMethods.Multiply:
                self._final_entity_scores = self._entity_scores * self._secondary_entity_scores
        else:
            self._final_entity_scores = self._entity_scores

        # and find the max score
        self._entity_predictions = tf.argmax(
            self._final_entity_scores,
            axis=1
        )
        self._entity_only_predictions = tf.argmax(
            self._entity_scores,
            axis=1
        )
        if self._p._secondary_entity_vocab_size > 0:
            self._secondary_entity_only_predictions = tf.argmax(
                self._secondary_entity_scores,
                axis=1
            )

        if debug:
            self._entity_predictions = tf.Print(self._entity_predictions, [self._entity_predictions], summarize=100, message="Entity Predictions")

def __test__():
    from params import LLParams
    params = LLParams(
        ctx_vocab_size = 4,
        ctx_dim = 3,
        entity_vocab_size = 4,
        entity_dim = 3,
        window_size = 1,
        max_num_entities = 2
    )

    ctx_embed_arr = np.array([
        [ 0.5, 0.3, 0.1 ],
        [ 0.8, 2.0, 1.3 ],
        [ 0.2, 0.4, 0.6 ],
        [ 1,   2,   3   ]
    ])
    entity_embed_arr = np.array([
        [ -0.1, -0.2, -0.3 ],
        [ 0.9, 0.9, 0.9 ],
        [ 1,   1,   1   ],
        [ 2,   0.5, 0   ]
    ])

    session = tf.Session()
    model = LinearSabbirLinkerB(session, ctx_embed_arr, entity_embed_arr, params, debug=True)

    batch_ctx_window_ixes = np.array([
        [ 1, 2 ],
        [ 3, 3 ]
    ])
    batch_ctx_window_masks = np.array([
        [ 1, 1 ],
        [ 1, 1 ]
    ])
    batch_entity_ixes = np.array([
        [ 0, 1 ],
        [ 3, 0 ]
    ])
    batch_entity_masks = np.array([
        [ 1, 1 ],
        [ 1, 1 ]
    ])

    ## Calculations:
    ##   <Sample 0 : W1, W2 | E0, E1>
    ##   <Sample 1 : W3, W3 | E3, E0>
    ##
    ## Mean ctx:
    ## [
    ##   [ 0.5  1.2  0.95 ]  #W1, W2
    ##   [ 1    2    3    ]  #W3, W3
    ## ]
    ##
    ## Entity cos sims
    ## [
    ##   [ -0.954   0.950 ]  # E0, E1 | W1, W2
    ##   [  0.389  -0.999 ]  # E3, E0 | W3, W3
    ## ]
    ##
    ## Projections
    ## [
    ##   [ 1.537 1.530 ]
    ##   [ 1.455 3.742 ]
    ## ]
    ##
    ## Scores
    ## [
    ##   [ -1.466  1.453 ]
    ##   [  0.566 -3.740 ]
    ## ]
    ##
    ## Predictions
    ## [
    ##   1  # E1
    ##   0  # E3
    ## ]

    (predictions, _) = model.getPredictions(batch_ctx_window_ixes, batch_ctx_window_masks, batch_entity_ixes, batch_entity_masks)
    __runTest__([1, 0], predictions)


    ## Calculations:
    ##   <Sample 0 : W1, -- | E0, E1>
    ##   <Sample 1 : W3, W3 | E3, E0>
    ##
    ## Mean ctx:
    ## [
    ##   [ 0.8  2    1.3  ]  #W1, --
    ##   [ 1    2    3    ]  #W3, W3
    ## ]
    ##
    ## Entity cos sims
    ## [
    ##   [ -0.924   0.941 ]  # E0, E1 | W1, W2
    ##   [  0.389  -0.999 ]  # E3, E0 | W3, W3
    ## ]
    ##
    ## Projections
    ## [
    ##   [ 2.325 2.367 ]
    ##   [ 1.455 3.742 ]
    ## ]
    ##
    ## Scores
    ## [
    ##   [ -2.148  2.227 ]
    ##   [  0.566 -3.739 ]
    ## ]
    ##
    ## Predictions
    ## [
    ##   1  # E1
    ##   0  # E3
    ## ]

    batch_ctx_window_masks = np.array([
        [ 1, 0 ],
        [ 1, 1 ]
    ])
    (predictions, _) = model.getPredictions(batch_ctx_window_ixes, batch_ctx_window_masks, batch_entity_ixes, batch_entity_masks)
    __runTest__([1, 0], predictions)

    batch_ctx_window_masks = np.array([
        [ 1, 1 ],
        [ 1, 1 ]
    ])
    batch_entity_masks = np.array([
        [ 1, 1 ],
        [ 1, 0 ]
    ])
    (predictions, _) = model.getPredictions(batch_ctx_window_ixes, batch_ctx_window_masks, batch_entity_ixes, batch_entity_masks)
    __runTest__([1, 0], predictions)

def __runTest__(expected, actual):
    print("\nExpected: [%s]" % (' '.join([str(v) for v in expected])))
    print("  Actual: [%s]" % (' '.join([str(v) for v in actual])))
    if np.array_equal(expected, actual):
        print("-- PASS --")
    else:
        print("-- FAIL --")

if __name__=='__main__':
    __test__()
