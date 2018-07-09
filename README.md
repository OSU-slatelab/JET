# JET (Jointly-embedded Entities and Text)

This is an open-source implementation of a method for jointly learning distributional embeddings of entities, words, and
terms from unlabeled text with distant supervision, described in the following paper:

- D Newman-Griffis, A M Lai, and E Fosler-Lussier, ["Jointly Embedding Entities and Text with Distant Supervision"](http://drgriffis.github.io/papers/2018-Repl4NLP.pdf).  In _Proceedings of the 3rd Workshop on Representation Learning for NLP_, 2018.

This work was also presented as a poster at the AMIA Informatics Summit 2018, titled "Jointly embedding biomedical entities and text with distant supervision."

## Overview

This repository contains three main components:

- `src` is the C implementation of the JET method, with all associated libraries.
- `preprocessing` is Python-based code for noisy annotation with a terminology.
- `experiments` is Python code for replicating the experiments found in the paper; for more information, please see [experiments/README](https://github.com/OSU-slatelab/JET/tree/master/experiments)

The included `demo.sh` script will download a tiny test corpus and run the preprocessing and JET implementations on it.

Pre-trained embeddings, along with other associated data from the paper, can be downloaded at [this link](https://slate.cse.ohio-state.edu/JET).

If you notice any issues with the code, please open up an issue in [the tracker](https://github.com/drgriffis/JET/issues)!

## Dependencies

The C code is self-contained; however, random behavior is implemented with a copy of Matsumoto and Nishimura's excellent implementation of Mersenne Twister (for more, see [their webpage](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html)), included in `src`.

The Python preprocessing code requires:

- NLTK

The experimental implementations require:

- [Tensorflow](http://www.tensorflow.org)
- NumPy/SciPy
- NLTK
- [pyemblib](https://github.com/drgriffis/pyemblib) (included)
- [configlogger](https://github.com/drgriffis/configlogger) (included)

## Reference

If you use this software/method in your own work, please cite the paper above:

```
@inproceedings(Newman-Griffis2018Repl4NLP,
  author = {Newman-Griffis, Denis and Lai, Albert M. and Fosler-Lussier, Eric},
  title = {Jointly Embedding Entities and Text with Distant Supervision},
  booktitle = {Proceedings of the 3rd Workshop on Representation Learning for NLP},
  year = {2018}
}
```
