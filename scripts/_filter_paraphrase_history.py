# Body of file-info callback (git-filter-repo wraps this in def callback(...)).
import json

if filename != b"datasets/chaosnli/items.jsonl":
    return (filename, mode, blob_id)
raw = value.get_contents_by_identifier(blob_id)
try:
    text = raw.decode("utf-8")
except UnicodeDecodeError:
    return (filename, mode, blob_id)
if b'"premise"' not in raw and b'"hypothesis"' not in raw:
    return (filename, mode, blob_id)
out_lines = []
for line in text.splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    if obj.get("role") == "paraphrase" or str(obj.get("id", "")).endswith("::paraphrase"):
        state = dict(obj.get("state") or {})
        state.pop("premise", None)
        state.pop("hypothesis", None)
        state["text_status"] = "local_paraphrase_required"
        obj["state"] = state
        out_lines.append(json.dumps(obj, sort_keys=True, ensure_ascii=False))
    else:
        out_lines.append(line)
new = ("\n".join(out_lines) + "\n").encode("utf-8")
return (filename, mode, value.insert_file_with_contents(new))
