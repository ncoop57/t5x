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

tfds_names = [
    "stablelm_2_tfds/stablelm-culturax-de:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk0:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk1:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk2:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk3:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk4:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk5:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk6:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk7:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk8:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk9:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk10:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk11:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk12:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk13:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-en-chunk14:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-es:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-fr:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-it:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-nl:1.0.0",
    "stablelm_2_tfds/stablelm-culturax-pt:1.0.0",
    "stablelm_2_tfds/stablelm-falcon-refinedweb:1.0.0",
    "stablelm_2_tfds/stablelm-fanfics-10k-10k:1.0.0",
    "stablelm_2_tfds/stablelm-openwebmath:1.0.0",
    "stablelm_2_tfds/stablelm-openwebtext:1.0.0",
    "stablelm_2_tfds/stablelm-pilev1-bookcorpus:1.0.0",
    "stablelm_2_tfds/stablelm-pilev1-hackernews:1.0.0",
    "stablelm_2_tfds/stablelm-pilev1-philpapers:1.0.0",
    "stablelm_2_tfds/stablelm-pilev2-amps:1.0.0",
    "stablelm_2_tfds/stablelm-pilev2-arxiv:1.0.0",
    "stablelm_2_tfds/stablelm-pilev2-dm_math:1.0.0",
    "stablelm_2_tfds/stablelm-pilev2-euro_parl:1.0.0",
    "stablelm_2_tfds/stablelm-pilev2-free_law:1.0.0",
    "stablelm_2_tfds/stablelm-pilev2-gutenberg:1.0.0",
    "stablelm_2_tfds/stablelm-pilev2-pile_of_law:1.0.0",
    "stablelm_2_tfds/stablelm-pilev2-pubmed:1.0.0",
    "stablelm_2_tfds/stablelm-pilev2-s2orc:1.0.0",
    "stablelm_2_tfds/stablelm-redpajama-c4:1.0.0",
    "stablelm_2_tfds/stablelm-redpajama-stackexchange:1.0.0",
    "stablelm_2_tfds/stablelm-starcoder-cleaned:1.0.0",
]
task_names = []
for name in tfds_names:
    task_name = f"c4_v220_span_corruption_{name.split('/')[1].split(':')[0]}"
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TfdsDataSource(tfds_name=name),
        preprocessors=[
            functools.partial(
                preprocessors.rekey, key_map={
                    "inputs": None,
                    "targets": "text"
                }),
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            preprocessors.span_corruption,
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[])

seqio.MixtureRegistry.add(
    "stable_t5_mixture",
    task_names,
    default_rate=seqio.mixing_rate_num_examples
)