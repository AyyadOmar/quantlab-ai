# Research website

The existing Next.js application and Vercel deployment structure are preserved. The homepage presents always-up and always-down baselines, study and model controls, per-asset comparisons, down-call accounting, uncertainty intervals, and a nine-stage research log. It is a published research snapshot, not a live prediction service.

The initial view is the newer-period retrospective confirmation (May 26–September 18, 2026), with logistic regression selected. It shows the full-history and validation-selected-history models beside both constant-direction baselines. Selected logistic regression scored 51.85%, versus 47.74% always-up and 52.26% always-down. The page explains that the ten extra correct predictions versus always-up all came from NVIDIA. Historical development studies remain separately selectable; scores from different periods are not treated as matched comparisons. Both model families remain available.

The research log records the unsuccessful intraday feature experiment and public news collection. News has no measured accuracy benefit yet, and Microsoft coverage was stale at the recorded collection. No predictive model or default trading setting was changed by this website update.

## Data and reports

`public/research/results.json` (schema version 2) contains only public summary fields for the newer confirmation, direction-rule and shared-company studies. It is generated from the saved research JSON reports, not retyped rounded values. Each study retains its own asset universe, evidence type, baselines, dates, and source hash. Always-down is derived as one minus the up-day share; flat sessions count as down. The homepage imports this small committed snapshot directly, so there are no client-side data fetches, fallback figures, or external API dependencies.

`public/research/reports/` contains 12 downloadable Markdown reports, including the evaluation protocol, newer confirmation, intraday experiment, diagnostics and news research plan. Reports describe the state when their experiments were completed; statements that the website was unchanged refer to that time. Relative links in the news report are adjusted for its published location. Legacy close-to-close snapshots remain in `public/demo/` and `docs/demo/` as archived references. Their charts, returns, and old sample signals are no longer shown as current results.

After completing new research, regenerate and check the snapshot from the project root:

```sh
.venv/bin/python scripts/export_website_research.py
npm run check:research
npm run build
```

Exporting requires the local research JSON outputs, which are intentionally excluded from Git. Cloning and building the website does not require those outputs: the compact public snapshot and downloadable reports are committed. Edit research-log conclusions when the actual evidence changes; never imply a snapshot is live.

The consistency check verifies 34 asset/model/scope rows, aggregate accounting, both baseline calculations, the newer-period reference counts and period separation, and all report downloads. The production build checks the page and metadata. The update was also checked in a browser: the visible results layout, model selection and switching between confirmation and historical studies. Responsive CSS retains wrapping controls and horizontally scrolling tables; no separate mobile viewport test was performed.

## Preview and sharing

```sh
npm run dev
```

Open the local address printed by Next.js. The homepage has no paid data or network fetch requirement. A branded Open Graph/X sharing image is in `public/og.png`; metadata resolves against Vercel's trusted deployment host environment variables, falling back to localhost for local previews.

## Publish through the existing repository

The user handles Git. This small follow-up can continue on the existing `news-research` branch, which already contains the research being published:

```sh
git status
git diff --stat
git add app/page.js app/globals.css public/research scripts/export_website_research.py scripts/check_website_research.mjs docs/website.md
git diff --cached --stat
git diff --cached
git commit -m "Clarify website baselines and research evidence"
git push origin news-research
```

Open a pull request into `main`. With the existing Vercel Git integration enabled, the branch push creates a preview deployment. Review that preview, then merge the pull request. If `main` is the configured production branch, merging publishes the update to the existing website. Verify the deployment reaches Ready in Vercel. Pushing only to a feature branch does not normally replace the production website.

To bring merged work back to this computer later, after committing or otherwise preserving local changes:

```sh
git switch main
git pull --ff-only origin main
```

`pull` downloads and integrates remote commits; it does not upload local edits. No pull is needed merely because local files were edited. Git's displayed ahead/behind status uses the last fetched remote state, so it is not a fresh check of GitHub until a fetch or pull happens.

This update does not create another hosting project, change domains, or replace the existing Vercel configuration. Actual production publication remains a separate GitHub/Vercel step after the website changes are committed.
