# Project website

This folder hosts the project page for *"Models That Know How Evaluations Are Designed Score Safer"*, served via GitHub Pages.

## Enable GitHub Pages

1. Push this folder to `main`.
2. On GitHub: **Settings → Pages**.
3. Under **Build and deployment → Source**, select **Deploy from a branch**.
4. Set **Branch** to `main` and folder to `/docs`. Save.
5. After a minute, the site will be live at:
   `https://compass-group-tue.github.io/arxiv2026_evaluation_meta_knowledge/`

## Local preview

The page is plain static HTML/CSS — no build step. Open `docs/index.html` directly in a browser, or serve the folder:

```bash
python -m http.server -d docs 8000
# then visit http://localhost:8000
```

## Files

- `index.html` — the page itself
- `static/css/style.css` — styles
- `static/images/teaser.png` — overview figure (converted from `paper/figures/optional_teaser.pdf`)
- `static/images/mcq_author_accuracy.png` — fictional-author MCQ figure
- `.nojekyll` — tells GitHub Pages to skip the Jekyll build and serve files as-is

