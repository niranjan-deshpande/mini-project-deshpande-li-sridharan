"""vLLM worker extension (--worker-extension-cls vllm_ext.AttnLoader): in-place load of merged attention weights.

vLLM's generic /collective_rpc reload_weights runs the layerwise reload machinery, which on gpt-oss fails to restore
the MXFP4 expert kernel tensors (decode then dies with "GptOssRoutedExperts has no attribute w13_weight"). This path
calls the model's ordinary load_weights on the attention tensors only: each param's weight_loader copies in place,
so tensor addresses (and captured CUDA graphs) stay valid and nothing else is touched."""
import torch


class AttnLoader:
    def load_attn(self, path: str) -> int:
        from safetensors.torch import load_file
        W = load_file(path, device="cpu")
        loaded = self.model_runner.get_model().load_weights(W.items())
        torch.cuda.synchronize()
        return len(loaded)
