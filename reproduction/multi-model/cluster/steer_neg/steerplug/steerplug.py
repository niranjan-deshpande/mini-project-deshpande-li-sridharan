"""vLLM general plugin: add a constant vector c to the residual stream after decoder layer STEER_LAYER of GLM-4.5-Air
(same as the HF forward hook in t3_steer.py: the layer returns (mlp_out, residual) and the stream is their sum, so
adding c to mlp_out adds c to the stream at every position). Off unless STEER_VEC (path to a .npy, already scaled)
is set."""
import os


def register():
    path = os.environ.get("STEER_VEC")
    if not path:
        return
    import numpy as np
    import torch
    from vllm.model_executor.models import glm4_moe as g
    cls = g.Glm4MoeDecoderLayer
    if getattr(cls, "_steer_patched", False):
        return
    L = int(os.environ["STEER_LAYER"])
    vec = np.load(path).astype(np.float32)
    init0, fwd0 = cls.__init__, cls.forward

    def init(self, *a, **k):
        init0(self, *a, **k)
        if self.layer_idx == L:
            self.register_buffer("steer_c", torch.tensor(vec, dtype=torch.get_default_dtype()), persistent=False)
            print(f"[steer] layer {L} |c|={np.linalg.norm(vec):.3f} device={self.steer_c.device} "
                  f"dtype={self.steer_c.dtype} pid={os.getpid()}", flush=True)

    def fwd(self, positions, hidden_states, residual):
        h, r = fwd0(self, positions, hidden_states, residual)
        if hasattr(self, "steer_c"):
            h = h + self.steer_c
        return h, r

    cls.__init__, cls.forward, cls._steer_patched = init, fwd, True
