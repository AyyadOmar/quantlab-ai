# Research website

The existing Next.js application and Vercel deployment structure are preserved. The homepage presents the completed research with an explicit always-up benchmark, study and model controls, per-asset comparisons, down-call accounting, uncertainty intervals, and a six-stage research log. It is a published historical snapshot, not a live prediction service.

## Data and reports

`public/research/results.json` contains only public summary fields for the direction-rule and shared-company studies. It is generated from the saved research JSON reports, not retyped rounded values. Each study retains its own asset universe, benchmark, dates, and source hash. The homepage imports this small committed snapshot directly, so there are no client-side data fetches, fallback figures, or external API dependencies.

`public/research/reports/` contains downloadable copies of the six Markdown study reports and evaluation protocol. Legacy close-to-close snapshots remain in `public/demo/` and `docs/demo/` as archived references. Their charts, returns, and old sample signals are no longer shown as current results.

After completing new research, regenerate and check the snapshot from the project root:

```sh
.venv/bin/python scripts/export_website_research.py
npm run check:research
npm run build
```

Exporting requires the local research JSON outputs, which are intentionally excluded from Git. Cloning and building the website does not require those outputs: the compact public snapshot and downloadable reports are committed. Edit research-log conclusions when the actual evidence changes; never imply a snapshot is live.

The consistency check verifies that aggregate scores reconcile with every asset's counts, down calls explain the difference from always-up, each scope covers the same dates and row counts recorded in its snapshot, and all report downloads exist. The production build checks the page and metadata. Responsive CSS supports narrow screens, keyboard focus, reduced motion, and horizontally scrolling tables; browser interaction/visual QA is separate from these checks.

## Preview and sharing

```sh
npm run dev
```

Open the local address printed by Next.js. The homepage has no paid data or network fetch requirement. A branded Open Graph/X sharing image is in `public/og.png`; metadata resolves against Vercel's trusted deployment host environment variables, falling back to localhost for local previews.

## Publish through the existing repository

The user handles Git. A dedicated `website-research-update` branch can be created from the current research branch so it includes the required research work:

```sh
git switch -c website-research-update
git add app public/research public/og.png scripts/export_website_research.py scripts/check_website_research.mjs package.json README.md docs/research_protocol.md docs/website.md
git diff --cached --stat
git commit -m "Publish current research and benchmark comparisons on website"
git push -u origin website-research-update
```

Open a pull request into `main`. With the existing Vercel Git integration enabled, the branch push creates a preview deployment. Review that preview, then merge the pull request. If `main` is the configured production branch, merging publishes the update to the existing website. Verify the deployment reaches Ready in Vercel. Pushing only to a feature branch does not normally replace the production website.

To bring merged work back to this computer later, after committing or otherwise preserving local changes:

```sh
git switch main
git pull --ff-only origin main
```

`pull` downloads and integrates remote commits; it does not upload local edits. No pull is needed merely because local files were edited. Git's displayed ahead/behind status uses the last fetched remote state, so it is not a fresh check of GitHub until a fetch or pull happens.

This update does not create another hosting project, change domains, or replace the existing Vercel configuration. Actual production publication remains a separate GitHub/Vercel step after the website changes are committed.
