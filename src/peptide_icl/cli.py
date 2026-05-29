import argparse
from pathlib import Path

from peptide_icl import ICLScorer


def _read_sequences_file(path: str) -> list[str]:
    """Read one sequence per line from a text file, ignoring blank lines."""
    seq_path = Path(path)
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {seq_path}")

    sequences: list[str] = []
    with seq_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            seq = line.strip().upper()
            if seq:
                sequences.append(seq)
    return sequences


def _write_scores_tsv(
    path: str, test_sequences: list[str], scores: list[float]
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("sequence\tscore\n")
        for seq, score in zip(test_sequences, scores):
            fh.write(f"{seq}\t{score:.6f}\n")


def _validate_equal_test_lengths(test_sequences: list[str]) -> int:
    if not test_sequences:
        raise ValueError("At least one test sequence is required.")

    expected_len = len(test_sequences[0])
    bad = [seq for seq in test_sequences if len(seq) != expected_len]
    if bad:
        lengths = sorted({len(seq) for seq in test_sequences})
        raise ValueError(
            "All test sequences must have the same length. "
            f"Found lengths: {lengths}."
        )
    return expected_len


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peptide-icl",
        description=(
            "Score test peptide sequences using in-context learning from "
            "positive and optional negative examples."
        ),
    )
    parser.add_argument(
        "--positive-file",
        required=True,
        help="Text file with one positive context sequence per line.",
    )
    parser.add_argument(
        "--negative-file",
        default=None,
        help="Optional text file with one negative context sequence per line.",
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Text file with one test/query sequence per line.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output TSV file path for sequence scores.",
    )
    parser.add_argument(
        "--esm-model",
        default="esm2_t6_8M_UR50D",
        help="ESM model name passed to fair-esm pretrained loader.",
    )
    parser.add_argument(
        "--linker",
        default="GGGGG",
        help="Linker inserted between context peptides when building prompts.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    positive_sequences = _read_sequences_file(args.positive_file)
    if args.negative_file is not None:
        negative_sequences = _read_sequences_file(args.negative_file)
    else:
        negative_sequences = None

    test_sequences = _read_sequences_file(args.test_file)

    if not positive_sequences:
        raise ValueError("At least one positive sequence is required.")

    query_len = _validate_equal_test_lengths(test_sequences)

    scorer = ICLScorer(
        positive_sequences=positive_sequences,
        negative_sequences=negative_sequences or None,
        esm_model=args.esm_model,
        linker=args.linker,
        query_len=query_len,
    )

    scores = scorer(test_sequences).detach().cpu().tolist()
    _write_scores_tsv(args.output_file, test_sequences, scores)


if __name__ == "__main__":
    main()
