def _resolve_ckpt(ckpt_path: str) -> str:
    """Resolve a checkpoint reference to a local file, downloading from the
    Hugging Face Hub when given an ``hf://[repo_id/]path/in/repo`` reference.
    """
    if not ckpt_path.startswith("hf://"):
        return ckpt_path
    from huggingface_hub import hf_hub_download
    ref = ckpt_path[len("hf://"):]
    parts = ref.split("/")
    repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
    assert repo_id == "therealgabeguo/BiB_generative", "only the default repo is supported"
    return hf_hub_download(repo_id=repo_id, filename=filename)
