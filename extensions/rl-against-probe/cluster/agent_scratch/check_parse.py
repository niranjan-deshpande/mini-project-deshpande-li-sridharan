from openai_harmony import Role
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
enc = get_encoding()
full = enc.encode("<|channel|>analysis<|message|>Think about it.\n\nMore.<|end|><|start|>assistant<|channel|>final<|message|>```python\nx=1\n```<|return|>", allowed_special="all")
def split(c_ids):
    try:
        msgs = enc.parse_messages_from_completion_tokens(c_ids, Role.ASSISTANT, strict=False)
    except Exception as e:
        return ("ERR", repr(e)[:120])
    ana = "".join(c.text for m in msgs if m.channel == "analysis" for c in m.content)
    fin = [c.text for m in msgs if m.channel == "final" for c in m.content]
    return ana, ("".join(fin) if fin else None)
print("full     ", split(full))
print("no_return", split(full[:-1]))
cut = full.index(200007)
print("trunc_ana", split(full[:cut - 1]))
# streaming parser fallback for truncated samples
from openai_harmony import StreamableParser
def split_stream(c_ids):
    p = StreamableParser(enc, role=Role.ASSISTANT)
    ana, fin = [], []
    for t in c_ids:
        p.process(t)
    for m in p.messages:
        (ana if m.channel == "analysis" else fin).append("".join(c.text for c in m.content))
    if p.current_channel == "analysis": ana.append(p.current_content)
    elif p.current_channel == "final": fin.append(p.current_content)
    return "".join(ana), ("".join(fin) if (fin or p.current_channel == "final") else None)
print("stream full", split_stream(full)); print("stream trunc", split_stream(full[:cut - 1]))
print("stream trunc_final", split_stream(full[:-3]))
