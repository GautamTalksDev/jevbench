# Arena replay data

Place JSON copied from scored `runs/` + `results/` here.

- `replay.json`primary payload the Arena loads in **Replay** mode
- Live mode never writes here or to scored run dirs; it only offers a DEMO download

Sync helper (from repo root):

```bash
python scripts/sync_arena_data.py
```
