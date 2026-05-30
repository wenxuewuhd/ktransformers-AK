"""Runtime patch: upstream kt_ep_wrapper expects ServerArgs.get_hf_config()."""

try:
    from sglang.srt.server_args import ServerArgs

    if not hasattr(ServerArgs, "get_hf_config"):

        def _get_hf_config(self):
            return self.get_model_config().hf_config

        ServerArgs.get_hf_config = _get_hf_config  # type: ignore[attr-defined]
except Exception:
    pass
