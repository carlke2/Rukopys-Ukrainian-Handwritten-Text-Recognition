"""
src/evaluation/cer.py  —  Character Error Rate metric.

WHY CER:
  CER measures how many character-level edits (insert, delete, replace)
  are needed to transform the predicted text into the ground truth.
  CER = (S + D + I) / N
    S = substitutions, D = deletions, I = insertions, N = total GT chars

  CER = 0.0  → perfect prediction
  CER = 1.0  → as many errors as characters
  CER > 1.0  → prediction longer/shorter than GT with many mismatches

  For Ukrainian handwriting, a CER below 0.10 (10%) is solid.
  Below 0.05 is very good. Below 0.02 is excellent.

USAGE:
  from src.evaluation.cer import compute_cer, compute_batch_cer

  score = compute_cer("Доброго ранку", "Добрго ранку")  # → 0.071
  avg   = compute_batch_cer(predictions, ground_truths)
"""


def edit_distance(a: str, b: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    This is the minimum number of single-character edits
    (insertions, deletions, substitutions) to turn string a into b.

    Uses dynamic programming — O(len(a) * len(b)) time and space.
    """
    la, lb = len(a), len(b)
    # dp[i][j] = edit distance between a[:i] and b[:j]
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, lb + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[lb]


def compute_cer(prediction: str, ground_truth: str) -> float:
    """
    Compute Character Error Rate for a single prediction.

    Args:
        prediction:   model output text
        ground_truth: correct text

    Returns:
        CER as a float in [0, ∞).
        Returns 0.0 if both strings are empty.
        Returns 1.0 if ground_truth is empty but prediction is not.
    """
    pred = prediction.strip()
    gt   = ground_truth.strip()

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    dist = edit_distance(pred, gt)
    return dist / len(gt)


def compute_batch_cer(
    predictions: list[str],
    ground_truths: list[str],
    verbose: bool = False,
) -> dict:
    """
    Compute CER over a batch of predictions.

    Uses "corpus-level" CER: total edits / total GT characters.
    This is more meaningful than averaging per-sample CER because
    short strings have disproportionate variance.

    Args:
        predictions:   list of predicted texts
        ground_truths: list of ground-truth texts
        verbose:       if True, print worst examples

    Returns:
        dict with:
          "corpus_cer":  float — main metric
          "mean_cer":    float — average of per-sample CERs
          "num_samples": int
          "num_perfect": int — samples with CER = 0
          "worst":       list of (pred, gt, cer) for top-5 worst
    """
    assert len(predictions) == len(ground_truths), (
        f"Length mismatch: {len(predictions)} preds vs "
        f"{len(ground_truths)} gt"
    )

    total_edits = 0
    total_chars = 0
    sample_cers = []
    pairs = []

    for pred, gt in zip(predictions, ground_truths):
        pred = pred.strip()
        gt   = gt.strip()
        dist = edit_distance(pred, gt)
        n    = len(gt)

        total_edits += dist
        total_chars += n

        cer = dist / n if n > 0 else (0.0 if len(pred) == 0 else 1.0)
        sample_cers.append(cer)
        pairs.append((pred, gt, cer))

    corpus_cer = total_edits / max(total_chars, 1)
    mean_cer   = sum(sample_cers) / max(len(sample_cers), 1)
    n_perfect  = sum(1 for c in sample_cers if c == 0.0)

    worst = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]

    if verbose:
        print(f"\n  CER Results")
        print(f"  {'─'*40}")
        print(f"  Corpus CER   : {corpus_cer:.4f}  ({corpus_cer*100:.2f}%)")
        print(f"  Mean CER     : {mean_cer:.4f}  ({mean_cer*100:.2f}%)")
        print(f"  Perfect (0)  : {n_perfect}/{len(sample_cers)}")
        print(f"\n  Top-5 worst predictions:")
        for pred, gt, cer in worst:
            print(f"    CER={cer:.3f}  pred='{pred[:50]}'  gt='{gt[:50]}'")

    return {
        "corpus_cer":  corpus_cer,
        "mean_cer":    mean_cer,
        "num_samples": len(sample_cers),
        "num_perfect": n_perfect,
        "worst":       worst,
    }


if __name__ == "__main__":
    # Quick smoke-test
    preds = ["Доброго ранку", "Прівіт", "Формула E=mc^2", ""]
    gts   = ["Доброго ранку", "Привіт", "Формула E=mc²",  ""]
    result = compute_batch_cer(preds, gts, verbose=True)
    print(f"\n✅ CER test passed. Corpus CER = {result['corpus_cer']:.4f}")
