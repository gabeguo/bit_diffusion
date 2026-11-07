"""Caption scorers shared by training eval and the offline COCO evaluator.

Legacy CIDEr splits raw strings on whitespace. COCO CIDEr and SPICE use the
official COCOEvalCap PTB tokenizer (case/punctuation normalization), then the
unchanged pycocoevalcap scorers. Score the whole corpus at once: CIDEr's IDF
depends on all reference images, not on generation batch size.
"""

import math
import shutil


def score_captions(references, predictions, *, legacy=True, cider=True, spice=True):
    """Score {image_id: [captions]} dictionaries; one prediction per image.

    Returns native scorer units. Multiply CIDEr/SPICE by 100 for the convention
    used in many captioning tables. Explicitly requested scorers fail loudly.
    Java 8 and pycocoevalcap are required for PTB/SPICE; SPICE also needs the
    Stanford models downloaded by its constructor on first use.
    """
    if not references or references.keys() != predictions.keys():
        raise ValueError("References and predictions must have identical, nonempty image IDs.")
    for key in references:
        if not references[key] or len(predictions[key]) != 1:
            raise ValueError(f"Expected >=1 references and exactly one prediction for {key}.")
        if not all(isinstance(s, str) for s in references[key] + predictions[key]):
            raise TypeError("Captions must be strings.")
    result = {}
    try:
        from pycocoevalcap.cider.cider import Cider
        if legacy:
            result["cider_legacy"] = float(Cider().compute_score(references, predictions)[0])
        if cider or spice:
            if shutil.which("java") is None:
                raise RuntimeError("Java is missing; install Java 8 and put java on PATH.")
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
            tokenizer = PTBTokenizer()
            refs = tokenizer.tokenize({k: [{"caption": s} for s in v] for k, v in references.items()})
            preds = tokenizer.tokenize({k: [{"caption": s} for s in v] for k, v in predictions.items()})
            if cider:
                result["cider_coco"] = float(Cider().compute_score(refs, preds)[0])
            if spice:
                from pycocoevalcap.spice.spice import Spice
                result["spice"] = float(Spice().compute_score(refs, preds)[0])
    except Exception as exc:
        raise RuntimeError(
            "Caption scoring failed. Install requirements-coco.txt and Java 8; "
            "SPICE needs its Stanford CoreNLP models and a writable package cache. "
            "See COCO_EVALUATION.md."
        ) from exc
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError(f"Caption scorer returned a non-finite result: {result}")
    return result
