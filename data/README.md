# Data

The repository does not contain a data set. Git ignores everything in `/data/` except this README.
The stereotype benchmarks contain offensive sentences on purpose. Read them with care.

| Source | Where | License or terms | Loader | Used for |
|---|---|---|---|---|
| CrowS-Pairs (Nangia et al., 2020) | `https://github.com/nyu-mll/crows-pairs` (`data/crows_pairs_anonymized.csv`) | CC BY-SA 4.0 | `load_crows_pairs` | Categories. Stereotype rows are biased |
| IndiBias (Sahoo et al., 2024) | `https://github.com/sahoonihar/IndiBias` | See the repository | `load_indibias` | Categories (adds `caste`) |
| SBIC, Social Bias Inference Corpus (Sap et al., 2020) | `https://maartensap.com/social-bias-frames/` | Research use, see the page | `load_sbic` | Biased and not-biased posts |
| Neutral sentences | Any CSV with a `text` column | Your own | `load_neutral` | Not-biased examples |

## Expected columns

| Loader | Columns |
|---|---|
| `load_crows_pairs` | `sent_more`, `bias_type`, `stereo_antistereo` |
| `load_indibias` | `modified_eng_sent_more`, `bias_type`, `stereo_antistereo`, optional `index` |
| `load_sbic` | `post`, `offensiveYN`, `targetCategory` (one row for each annotation) |
| `load_neutral` | `text` |

## Prepare a unified file

```bash
biasradar prepare --crows data/crows_pairs.csv --indibias data/IndiBias_v1_sample.csv \
    --sbic data/SBIC/SBIC.v2.trn.csv data/SBIC/SBIC.v2.dev.csv --out data/unified.csv
```

The output has the columns `text, source, group_id, is_biased, categories, polarity, split`.
SBIC posts can contain user handles and offensive language. Do not commit them.

## Synthetic data (no download)

`biasradar synth --out data/synthetic.csv` writes a template corpus with mild generalisations and neutral sentences.
The tests and the demo use only this synthetic data.
