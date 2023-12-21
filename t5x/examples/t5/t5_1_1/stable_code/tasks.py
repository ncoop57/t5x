# ${HOME}/dir1/user_dir/tasks.py

import functools
import seqio
import t5
import tensorflow_datasets as tfds
from t5.evaluation import metrics
from t5.data import preprocessors

vocabulary = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model', extra_ids=100)
output_features = {
    'inputs': seqio.Feature(vocabulary=vocabulary),
    'targets': seqio.Feature(vocabulary=vocabulary)
}

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=vocabulary, add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=vocabulary, add_eos=True)
}

seqio.TaskRegistry.add(
    "sc_lm",
    source=seqio.TfdsDataSource(tfds_name="starcoder/lm:1.0.0"),
    preprocessors=[
        # functools.partial(
        #     preprocessors.rekey, key_map={
        #         "inputs": None,
        #         "targets": "text"
        #     }),
        preprocessors.lm,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

seqio.TaskRegistry.add(
    "sc_fim",
    source=seqio.TfdsDataSource(tfds_name="starcoder/lm:1.0.0"),
    preprocessors=[
        # functools.partial(
        #     preprocessors.rekey, key_map={
        #         "inputs": None,
        #         "targets": "text"
        #     }),
        preprocessors.fill_in_the_blank,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        # seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


seqio.MixtureRegistry.add(
    "stable_code_mixture",
    ["sc_lm", "sc_fim"],
    default_rate=1.0)