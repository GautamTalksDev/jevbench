# Publishing the certificate arena

**Do not share the old claude.ai public link.** The harness verdict wiring
changed (Amendment 9); that hosted hash still uses the interval-based verdict
and is stale.

## Same-commit rule

The page and the paper must come from the **same git commit**. Prefer hosting
`arena/` from this repo (GitHub Pages). Until a remote exists, serve locally:

```bash
python3 scripts/sync_arena_data.py
cd arena && python3 -m http.server 8765 --bind 127.0.0.1
# open http://127.0.0.1:8765/
```

## GitHub Pages (once `origin` points at GitHub)

1. Push this branch.
2. Repo → Settings → Pages → Source: **GitHub Actions**.
3. The workflow `.github/workflows/pages.yml` publishes the `arena/` folder
   on every push to `main`.
4. Certificate data: run `jevbench export-certificate` (or sync specimen) into
   `arena/data/` before the push so the page matches the commit.

## Hand-off

If Pages is not ready, send the reviewer the `arena/` tree from the binding
commit (zip or raw files), never the stale claude.ai URL.
