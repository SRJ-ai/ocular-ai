# OcularAI XAI Enhancement Design
**Date:** 2026-04-18  
**Status:** Approved  
**Target:** Medical/clinical professionals

---

## Overview

Add an Explainable AI (XAI) section to the OcularAI retinal fundus classifier using **Occlusion Sensitivity** (Option A) — a model-agnostic approach compatible with the Teachable Machine MobileNet export format.

Three compact sub-features, all inside a single collapsible EXPLAINABILITY accordion:
1. **Heatmap overlay** — semi-transparent red/blue canvas drawn over the preview image
2. **Region contribution bars** — 4 retinal quadrant scores (Superior/Inferior/Nasal/Temporal)
3. **Counterfactual rationale** — one static clinical sentence per predicted class

No new files. All changes land in `index.html`, `css/style.css`, `js/app.js`.  
CSS bumps to `v11`, JS to `v10`.

---

## Section 1 — Architecture & Data Flow

```
runPrediction()
  → showResults()          ← report renders immediately
  → runXAI(img, topIdx)    ← async, fire-and-forget

runXAI:
  1. 8×8 grid = 64 patches, each 28×28px
  2. For each patch: blackout → resize to 224×224 → model.predict() → record Δconfidence
  3. Normalize 64 scores → red/blue colormap → draw semi-transparent canvas over #preview-img
  4. Aggregate 4 quadrants (top-left=Superior, bottom-left=Inferior,
     top-right=Nasal, bottom-right=Temporal) → render region bars
  5. Look up static rationale text for topClass → render <p>
```

- All tensor allocations wrapped in `tf.tidy()` — no manual disposal
- ~0.5s total on desktop; "Computing explanation…" spinner shown while running
- Runs after `showResults()` so the report is never blocked

---

## Section 2 — UI Layout

**EXPLAINABILITY accordion** — last row inside `#results-output`, collapsed by default:

```
▶ EXPLAINABILITY                    [toggle arrow]
─────────────────────────────────────────────────
  [Computing explanation…]   ← spinner during XAI

  After XAI completes:

  HEATMAP
  ┌─────────────────────────────────────────┐
  │  existing preview image + heat overlay  │
  │  (canvas drawn on top, semi-transparent)│
  └─────────────────────────────────────────┘
  🔴 High influence  ○ Neutral  🔵 Low influence

  REGION CONTRIBUTION
  Superior   ████████░░  72%
  Inferior   █████░░░░░  48%
  Nasal      ██████████  91%
  Temporal   ███░░░░░░░  30%

  RATIONALE
  "Optic disc pallor and increased cup-to-disc ratio
   in the nasal region are primary indicators of
   Glaucoma in this scan."
```

**Implementation notes:**
- Accordion header is a `<button>` — zero DOM footprint when collapsed
- Heatmap canvas sized to match `#preview-img` dimensions (responsive, redrawn on load)
- Region bars reuse existing `.conf-row` / `.conf-fill` CSS — no new bar classes needed
- Rationale is a single `<p>` — one sentence per class in a JS object literal

**New HTML IDs:**
- `#xai-section` — accordion wrapper
- `#xai-toggle` — header button
- `#xai-body` — collapsible content area
- `#xai-spinner` — "Computing explanation…" state
- `#xai-content` — shown after XAI completes
- `#heatmap-canvas` — overlay canvas (positioned absolute over `#preview-img`)
- `#xai-region-list` — region contribution bar container
- `#xai-rationale` — rationale `<p>`

---

## Section 3 — Error Handling & Edge Cases

| Scenario | Behaviour |
|---|---|
| XAI takes > 3s | Spinner stays visible; no timeout — mobile CPUs may be slow |
| `runXAI` throws (tensor OOM, etc.) | Accordion shows "Explanation unavailable" — report fully usable |
| User clicks "Analyze Next Image" mid-XAI | `runXAI` is fire-and-forget; abandoned silently, old canvas discarded |
| `restoreScan` from history | XAI section hidden — re-running XAI on restored scans is out of scope |
| Image < 56px on either axis | Patch grid still runs; scores may be noisy but won't crash |
| Heatmap canvas resize | Redrawn on `#preview-img` load event each time |

---

## Files Changed

| File | Scope of change |
|------|----------------|
| `index.html` | +XAI accordion HTML (`#xai-section` and children) |
| `css/style.css` | +accordion styles, +heatmap canvas positioning, +XAI spinner |
| `js/app.js` | +`runXAI()`, +`drawHeatmap()`, +`renderRegionBars()`, +rationale lookup, update `showResults()` and `clearImage()` |

**CSS version bump:** `?v=10` → `?v=11`  
**JS version bump:** `?v=9` → `?v=10`

---

## Out of Scope

- True Grad-CAM (requires intermediate layer access not available in TM export)
- Re-running XAI on history-restored scans
- Saving heatmap to PDF export (future enhancement)
- Per-patch confidence values in tooltip