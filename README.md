# Peptide ICL

Peptide ICL is a tool for in-context learning of peptide sequences.  The tool uses in-context learning in the ESM model to build a fast, efficient screening tool for short protein sequences.

## Installation
To install Peptide ICL, you can use pip.  First, clone the repository and navigate to the source directory:

```bash
    git clone https://github.com/thiedelab/peptide_icl.git
    cd peptide-icl
```

Then, install the package.

```bash
    pip install .
```

(You can also install in editable mode with `pip install -e .` if you want to make changes to the code.)

## Usage

To use Peptide ICL, you can import the `ICLScorer` class and create an instance.  The scorer takes a list of positive sequences (sequences with a desired property) as well as an optional list of negative sequences (sequences without the desired property).  You can then call the scorer with a list of query peptides to get scores for each query. 

```python
from peptide_icl import ICLScorer

...

icl_scorer = ICLScorer(positive_sequences, negative_sequences, query_len=15)
scores = icl_scorer(query_peptides)
```

A higher score indicates that the query peptide is more similar to the positive sequences and less similar to the negative sequences.
A more detailed example can be found in the `examples/mhc` directory of the repository.

## Command Line Interface

After installation, you can score peptides from the command line:

```bash
peptide-icl \
    --positive-file data/positive.txt \
    --negative-file data/negative.txt \
    --test-file data/test.txt \
    --output-file results/scores.tsv \
    --esm-model esm2_t6_8M_UR50D
```

Each input file should be raw text with one peptide sequence per line.
The output is written as a tab-separated file with columns: `sequence` and `score`.

Important: all sequences passed via `--test-file` must have the same length. The CLI raises a `ValueError` if lengths differ.
