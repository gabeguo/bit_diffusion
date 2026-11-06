"""CPU integration checks for the fixed-vocabulary ablation and existing loader."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from coco_eval import encode_prompts
from data_utils.latent_dataset import CommonCatalogLatentDataset
from data_utils.prepare_random_token_data import prepare
from token_bridge import BRIDGE_RUNTIME_PRESETS, bridge_to_token_flat


def make_source(root, preset="sd"):
    root.mkdir()
    bc = BRIDGE_RUNTIME_PRESETS[preset].bridge
    cfg = {"shard_size": 2, "text_model": "test-tokenizer", "bridge_preset": preset}
    (root / "config.json").write_text(json.dumps({"config": cfg}))
    cfg.update(token_seq_len=bc.token_seq_len, token_emb_dim=128, mrl_dim=128,
               token_emb_storage="table", vocab_table="token_vocab_table.npy")
    (root / "token_embed_config.json").write_text(json.dumps({"config": cfg}))
    np.save(root / "token_vocab_table.npy", np.ones((2048, 128), dtype=np.float16))
    np.zeros((2, *bc.bridge_shape), dtype=np.float16).tofile(root / "latents_00000.memmap")
    np.ones(2, dtype=np.uint8).tofile(root / "filled_00000.bin")
    tokens = root / "token_original"
    tokens.mkdir()
    ids = np.zeros((2, bc.token_seq_len), dtype=np.int32)
    ids[:, :3] = [7, 3, 7]
    mask = np.zeros_like(ids, dtype=np.uint8)
    mask[:, :3] = 1
    for suffix, values in (("token_ids", ids), ("mask", mask),
                           ("lengths", np.full(2, 3, dtype=np.uint16)),
                           ("filled", np.ones(2, dtype=np.uint8))):
        values.tofile(tokens / f"token_original_{suffix}_00000.memmap")


def token_array(batch):
    # Use ordinary pipe serialization, avoiding platform-specific tensor
    # shared-memory transports; the dataset still runs in separate workers.
    return batch[0]["text_token_emb"].numpy()


class RandomTokenDataTests(unittest.TestCase):
    def test_same_token_vectors_across_splits_workers_and_coco_encoding(self):
        for preset in ("sd", "flux"):
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                train, test, output = root / "train", root / "test", root / "random"
                make_source(train, preset)
                make_source(test, preset)
                # Older held-out sets may store dense token embeddings. Only
                # their IDs/masks are needed when using the new shared table.
                if preset == "sd":
                    manifest = test / "token_embed_config.json"
                    cfg = json.loads(manifest.read_text())
                    cfg["config"]["token_emb_storage"] = "memmap"
                    manifest.write_text(json.dumps(cfg))
                    (test / "token_vocab_table.npy").unlink()
                prepare(train, test, output, seed=17)
                runtime = BRIDGE_RUNTIME_PRESETS[preset]
                bc = runtime.bridge
                datasets = [CommonCatalogLatentDataset(output / split, cast_dtype=torch.float32,
                                                       token_pad_id=2) for split in ("train", "test")]
                a, b = datasets[0][0], datasets[1][0]
                self.assertTrue(torch.equal(a["text_token_emb"], b["text_token_emb"]))
                vectors = a["text_token_emb"].reshape(bc.token_seq_len, bc.token_emb_dim)
                self.assertTrue(torch.equal(vectors[0], vectors[2]))
                self.assertFalse(torch.equal(vectors[0], vectors[1]))
                torch.testing.assert_close(vectors[:3].norm(dim=-1), torch.full((3,), runtime.token_scale))
                self.assertEqual(vectors[3:].count_nonzero().item(), 0)
                self.assertTrue((a["text_token_ids"][3:] == 2).all())
                for batch in DataLoader(datasets[0], batch_size=1, num_workers=2,
                                        collate_fn=token_array, timeout=10):
                    self.assertTrue(np.array_equal(batch, a["text_token_emb"].numpy()))
                table = np.load(output / "token_vocab_table.npy")
                self.assertAlmostEqual(float(table.mean()), 0, delta=0.01)
                self.assertAlmostEqual(float(table.std()), 1, delta=0.01)
                def tokenize(captions, **kwargs):
                    return {"input_ids": a["text_token_ids"].numpy()[None],
                            "attention_mask": a["text_token_mask"].numpy()[None]}
                encoded = encode_prompts(["ignored"], tokenize, table, runtime, "cpu")
                torch.testing.assert_close(bridge_to_token_flat(encoded, config=bc)[0], a["text_token_emb"])
                self.assertTrue((output / "train/latents_00000.memmap").is_symlink())
                self.assertTrue(np.all(np.load(train / "token_vocab_table.npy") == 1))

    def test_reproducibility_and_incompatible_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = root / "train", root / "test"
            make_source(train)
            make_source(test)
            prepare(train, test, root / "one", 0)
            prepare(train, test, root / "one", 0)
            prepare(train, test, root / "two", 0)
            prepare(train, test, root / "three", 1)
            table = lambda name: np.load(root / name / "token_vocab_table.npy")
            self.assertTrue(np.array_equal(table("one"), table("two")))
            self.assertFalse(np.array_equal(table("one"), table("three")))
            with self.assertRaisesRegex(ValueError, "fresh --out-dir"):
                prepare(train, test, root / "one", 1)
            with self.assertRaisesRegex(ValueError, "separate"):
                prepare(train, test, train / "random", 0)
            np.save(root / "one/token_vocab_table.npy", np.zeros((2048, 128)))
            with self.assertRaisesRegex(ValueError, "fresh --out-dir"):
                prepare(train, test, root / "one", 0)
            cfg_path = test / "token_embed_config.json"
            cfg = json.loads(cfg_path.read_text())
            cfg["config"]["text_model"] = "different-tokenizer"
            cfg_path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "same tokenizer"):
                prepare(train, test, root / "four", 0)


if __name__ == "__main__":
    unittest.main()
