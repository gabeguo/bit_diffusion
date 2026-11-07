import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from caption_metrics import score_captions
from coco_eval import encode_prompts, select_samples, validate_outputs, write_json
from data_utils.download_coco import download, extract
from token_bridge import BRIDGE_RUNTIME_PRESETS, bridge_to_token_flat


class CocoTests(unittest.TestCase):
    def test_selection_uses_caption_ids_and_allows_repeated_image_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "annotations").mkdir()
            annotations = {"images": [{"id": i, "file_name": f"{i}.jpg"} for i in range(2)],
                           "annotations": [{"id": i, "image_id": i // 5, "caption": str(i)} for i in range(10)]}
            write_json(root / "annotations/captions_val2014.json", annotations)
            a = select_samples(root, "t2i", 10, 7)
            self.assertEqual(a, select_samples(root, "t2i", 10, 7))
            self.assertEqual(len({r["caption_id"] for r in a}), 10)
            self.assertEqual(len({r["image_id"] for r in a}), 2)

    def test_prompt_encoding_scales_vectors_and_zeroes_padding(self):
        runtime = BRIDGE_RUNTIME_PRESETS["sd"]
        table = np.ones((3, 128), dtype=np.float32)
        def tokenizer(captions, **kwargs):
            mask = np.zeros((1, 64), dtype=np.int64)
            mask[:, :2] = 1
            return {"input_ids": np.ones((1, 64), dtype=np.int64), "attention_mask": mask}
        encoded = encode_prompts(["hello"], tokenizer, table, runtime, "cpu")
        tokens = bridge_to_token_flat(encoded).reshape(1, 64, 64)
        self.assertTrue(torch.allclose(tokens[:, :2], torch.ones(1, 2, 64)))
        self.assertEqual(tokens[:, 2:].count_nonzero().item(), 0)

    def test_rejects_missing_or_mixed_shards_and_duplicate_captions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = {"world_size": 2, "task": "i2t", "fingerprint": "same",
                        "samples": [{"index": i, "image_id": i} for i in range(3)]}
            for rank in range(2):
                directory = root / f"rank-{rank:05d}"
                directory.mkdir()
                indices = list(range(3))[rank::2]
                write_json(directory / "done.json", {"fingerprint": "same", "indices": indices})
                write_json(directory / "predictions.json", [{"image_id": i, "caption": "a dog"} for i in indices])
            self.assertEqual(len(validate_outputs(root, protocol)), 3)
            write_json(root / "rank-00001/predictions.json", [{"image_id": 0, "caption": "duplicate"}])
            with self.assertRaises(ValueError):
                validate_outputs(root, protocol)
            (root / "rank-00001/done.json").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_outputs(root, protocol)

    def test_download_resume_and_safe_extraction(self):
        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Length": "3", "Content-Range": "bytes 3-5/6"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.zip"
            path.with_suffix(".zip.part").write_bytes(b"abc")
            with patch("urllib.request.urlopen", return_value=Response(b"def")) as request:
                download("https://example.org/file.zip", path)
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertEqual(request.call_args.args[0].headers["Range"], "bytes=3-")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../outside.txt", "unsafe")
            with self.assertRaises(ValueError):
                extract(path, Path(tmp) / "extract")

    @unittest.skipUnless(os.environ.get("RUN_COCO_METRICS") == "1", "Requires pycocoevalcap, Java and SPICE models")
    def test_standard_scorers_match_coco_pipeline(self):
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        refs = {1: ["A red ball on a table.", "The ball is red."],
                2: ["A dog runs on grass.", "A dog is running outside."],
                3: ["Two birds sit in a tree.", "Birds are in the tree."]}
        preds = {key: [values[0].upper()] for key, values in refs.items()}
        actual = score_captions(refs, preds)
        tokenizer = PTBTokenizer()
        tokenize = lambda data: tokenizer.tokenize({k: [{"caption": s} for s in v] for k, v in data.items()})
        r, p = tokenize(refs), tokenize(preds)
        self.assertAlmostEqual(actual["cider_coco"], Cider().compute_score(r, p)[0])
        self.assertAlmostEqual(actual["spice"], Spice().compute_score(r, p)[0])
        self.assertNotEqual(actual["cider_legacy"], actual["cider_coco"])
        self.assertGreater(actual["spice"], 0)


if __name__ == "__main__":
    unittest.main()
