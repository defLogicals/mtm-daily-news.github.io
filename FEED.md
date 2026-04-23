# Marathi Maharashtra app feed (static JSON)

This project builds a **Marathi Maharashtra digest**: news, jobs, exams, law and policy that matters to **Marathi-speaking readers in Maharashtra** (local, cultural, and civic focus). Pan-India English feeds are intentionally minimal; national items (e.g. PIB, DD, PRS) are kept only when crawl-time filters show Maharashtra or Marathi relevance.

After each GitHub Actions run or local `npm test`, the site under `site/` is deployed to GitHub Pages.

## Base URL

Replace with your live Pages URL, for example:

`https://<user-or-org>.github.io/mtm-daily-news.github.io/` or the root URL shown in **Settings → Pages**.

## Endpoints

| File | Purpose |
|------|---------|
| `site/data.json` | Primary payload: full processed set used by the static UI (rolling window). |
| `site/dates.json` | `availableDates`: list of `YYYY-MM-DD` keys with per-day snapshots. |
| `site/data/YYYY-MM-DD.json` | Articles for that calendar day (copy of `data/<date>-processed.json`). |

## Shape (simplified)

Top-level keys typically include:

- `articles`: array of items with `title`, `url`, `source`, `source_domain`, `source_category`, `pubDate`, `category`, `difficulty`, `confidence`, `summary`, `entities`, `id`, etc.
- `processedAt`, `totalArticles`, `categories` (where emitted by the pipeline).

## Scrolling in the app

1. Load `data.json` for the newest combined list (or a known date file for “archive” mode).
2. For “load older,” read `dates.json`, then request `data/<previous-date>.json` sequentially.

## Testing without the app

Open `site/index.html` via `npm run dev` (static server) to verify the same JSON-driven content in a browser.

Disclaimer: discovery only; not legal, tax, or career advice.

## Sources configuration (`sources.json`)

- **`sources`**: Marathi + Maharashtra regional outlets, limited city-scoped English (Mumbai/Pune), and a small set of national legal/policy RSS (PIB, DD, PRS) filtered at crawl time for MH relevance.
- **`medium_blogs`**, **`article_sources`** (Reddit: MH cities only), **`youtube_channels`**, **`newsletters`** (optional), **`developer_blogs`** (optional), **`academic_sources`**, **`job_opportunity`**: merged into the same crawl as `sources` (shape: `name`, `url`, `category`, `priority`).
- **`priority`**: `highest` (Marathi + core discussion feeds) → `high` → `medium` → `low`. Crawl order follows this so Marathi-heavy feeds run first in each batch.

## Jobs & LinkedIn

- **LinkedIn** does not offer a stable, public RSS for job searches or company updates that fits this repo’s RSS-only crawler. To use LinkedIn you would need their **official API** (partnership / compliance) or a separate backend—not something we can add as a feed URL here.
- **Job portals**: Where an RSS exists, it is included (e.g. ET Jobs, Freshersvoice, Pagalguy). Items are filtered toward Maharashtra relevance or widely taken exams (MPSC, SSC, etc.). **Indeed** and some others often return **403/404 to automated clients**; if a URL fails in Actions, replace it in `sources.json` or use another portal’s feed.

## Marathi & local

- Titles are filtered with a **Maharashtra / Marathi reader** relevance model plus keyword fallbacks (Latin transliterations and place names). Trusted Marathi and MH-local sources skip the strict title gate; PIB/DD and other national feeds require MH hooks in the headline or description keyword path.
