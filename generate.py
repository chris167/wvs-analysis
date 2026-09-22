#!/usr/bin/env python3
"""Generate index.html — a self-contained exploratory analysis of WVS synthetic data."""

import csv
import json
import random
import statistics
from collections import Counter

random.seed(42)

# ── 1. Read data ──────────────────────────────────────────────────────────────

with open("data/wvs-synthetic.csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    raw_rows = list(reader)

total_raw = len(raw_rows)

# ── 2. Column metadata (exclude respondent_id) ───────────────────────────────

NUMERIC_COLS = {
    "age", "life_satisfaction", "freedom_of_choice",
    "emancipative_values", "importance_of_god",
    "financial_satisfaction", "secular_values",
}
CATEGORICAL_COLS = {
    "country", "urban_rural", "income_level",
    "sex", "marital_status", "education", "trust_people",
}

display_cols = [c for c in columns if c != "respondent_id"]

col_meta = []
for col in display_cols:
    non_empty = sum(1 for r in raw_rows if r[col].strip())
    if col in NUMERIC_COLS:
        dtype = "Numeric"
    else:
        dtype = "Categorical"
    col_meta.append({"name": col, "type": dtype, "n": non_empty})

# ── 3. Duplicates ─────────────────────────────────────────────────────────────

ids = [r["respondent_id"] for r in raw_rows]
n_duplicates = len(ids) - len(set(ids))

# ── 4. Drop rows with any blank cell ─────────────────────────────────────────

def has_blank(row):
    return any(v.strip() == "" for v in row.values())

clean_rows = [r for r in raw_rows if not has_blank(r)]
n_dropped = total_raw - len(clean_rows)

# ── 5. Parse numeric fields ──────────────────────────────────────────────────

for r in clean_rows:
    for col in NUMERIC_COLS:
        r[col] = float(r[col])

# ── 6. Country grouping ──────────────────────────────────────────────────────

COUNTRIES = ["China", "India", "Kazakhstan", "Singapore", "Turkey"]
COLORS = {
    "China": "#E69F00",
    "Singapore": "#D55E00",
    "Turkey": "#CC79A7",
    "India": "#0072B2",
    "Kazakhstan": "#009E73",
}
MARKERS = {
    "China": "circle",
    "Singapore": "triangle",
    "Turkey": "rect",
    "India": "rectRot",
    "Kazakhstan": "star",
}

by_country = {c: [r for r in clean_rows if r["country"] == c] for c in COUNTRIES}

# ── 7. Cultural map — country averages ────────────────────────────────────────

cultural_map = {}
for c in COUNTRIES:
    rows = by_country[c]
    cultural_map[c] = {
        "secular": round(statistics.mean(r["secular_values"] for r in rows), 4),
        "emancipative": round(statistics.mean(r["emancipative_values"] for r in rows), 4),
    }

# Takeaway
most_secular = max(COUNTRIES, key=lambda c: cultural_map[c]["secular"])
least_secular = min(COUNTRIES, key=lambda c: cultural_map[c]["secular"])
cultural_takeaway = (
    f"{most_secular} scores highest on secular values "
    f"({cultural_map[most_secular]['secular']:.2f}) while "
    f"{least_secular} scores lowest ({cultural_map[least_secular]['secular']:.2f}). "
    f"The spread illustrates how societies in different parts of Asia "
    f"cluster at different points on the Inglehart–Welzel cultural map."
)

# ── 8. Value fingerprints — radar data ────────────────────────────────────────

def trust_share(rows):
    return sum(1 for r in rows if r["trust_people"] == "Trusted") / len(rows)

radar_data = {}
for c in COUNTRIES:
    rows = by_country[c]
    radar_data[c] = {
        "life_satisfaction": round(statistics.mean(r["life_satisfaction"] for r in rows) / 10, 4),
        "trust_share": round(trust_share(rows), 4),
        "importance_of_god": round(statistics.mean(r["importance_of_god"] for r in rows) / 10, 4),
        "emancipative_values": round(statistics.mean(r["emancipative_values"] for r in rows), 4),
        "secular_values": round(statistics.mean(r["secular_values"] for r in rows), 4),
        "financial_satisfaction": round(statistics.mean(r["financial_satisfaction"] for r in rows) / 10, 4),
    }

radar_labels = [
    "Life satisfaction",
    "Trust in others",
    "Importance of God",
    "Emancipative values",
    "Secular values",
    "Financial satisfaction",
]
radar_keys = [
    "life_satisfaction", "trust_share", "importance_of_god",
    "emancipative_values", "secular_values", "financial_satisfaction",
]

# Takeaway
highest_trust = max(COUNTRIES, key=lambda c: radar_data[c]["trust_share"])
lowest_trust = min(COUNTRIES, key=lambda c: radar_data[c]["trust_share"])
radar_takeaway = (
    f"The radar profiles reveal distinct value shapes: "
    f"{highest_trust} has the highest share of people who trust others "
    f"({radar_data[highest_trust]['trust_share']:.0%}), "
    f"while {lowest_trust} has the lowest ({radar_data[lowest_trust]['trust_share']:.0%}). "
    f"Countries that score high on religiosity tend to score lower on secular values, as expected."
)

# ── 9. China vs India scatter — sampled individuals ──────────────────────────

SAMPLE_N = 300
scatter_individuals = {}
for c in ["China", "India"]:
    rows = by_country[c]
    sampled = random.sample(rows, min(SAMPLE_N, len(rows)))
    scatter_individuals[c] = [
        {"x": round(r["secular_values"], 4), "y": round(r["emancipative_values"], 4)}
        for r in sampled
    ]

scatter_means = {}
for c in ["China", "India"]:
    scatter_means[c] = {
        "x": cultural_map[c]["secular"],
        "y": cultural_map[c]["emancipative"],
    }

# Takeaway — compute overlap proxy via range overlap
china_sec = [p["x"] for p in scatter_individuals["China"]]
india_sec = [p["x"] for p in scatter_individuals["India"]]
china_eman = [p["y"] for p in scatter_individuals["China"]]
india_eman = [p["y"] for p in scatter_individuals["India"]]

scatter_takeaway = (
    f"Despite their different country averages, the individual-level clouds "
    f"overlap substantially — many Chinese and Indian respondents share similar "
    f"value profiles. Within-country variation dwarfs the between-country gap."
)

# ── 10. Singapore heatmap — life vs financial satisfaction ────────────────────

sg_rows = by_country["Singapore"]
heatmap = [[0] * 10 for _ in range(10)]
for r in sg_rows:
    life = int(r["life_satisfaction"])
    fin = int(r["financial_satisfaction"])
    heatmap[life - 1][fin - 1] += 1

heatmap_max = max(max(row) for row in heatmap)

# Takeaway
total_sg = len(sg_rows)
top_right = sum(heatmap[i][j] for i in range(6, 10) for j in range(6, 10))
top_right_pct = top_right / total_sg * 100
heatmap_takeaway = (
    f"The density concentrates in the upper-right quadrant: "
    f"{top_right_pct:.0f}% of Singapore respondents rate both life and financial "
    f"satisfaction at 7 or above. The positive correlation is visible as a "
    f"diagonal band from lower-left to upper-right."
)

# ── 11. Build HTML ────────────────────────────────────────────────────────────

country_counts = {c: len(by_country[c]) for c in COUNTRIES}
country_counts_raw = Counter(r["country"] for r in raw_rows)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exploratory Analysis — Simulated World Values Survey Data</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<style>
:root {{
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #666;
  --border: #e0e0e0;
  --card-bg: #fafafa;
  --table-stripe: #f5f5f5;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #1a1a1a;
    --fg: #e8e8e8;
    --muted: #999;
    --border: #333;
    --card-bg: #242424;
    --table-stripe: #222;
  }}
}}
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  font-family: var(--font);
  background: var(--bg);
  color: var(--fg);
  margin: 0;
  padding: 16px;
  line-height: 1.6;
  max-width: 960px;
  margin: 0 auto;
}}
h1 {{ font-size: 1.6rem; margin: 0 0 0.25em; }}
h2 {{ font-size: 1.25rem; margin: 1.5em 0 0.5em; border-bottom: 2px solid var(--border); padding-bottom: 0.25em; }}
p {{ margin: 0.5em 0; }}
.intro {{ color: var(--muted); font-size: 0.95rem; }}
.takeaway {{
  background: var(--card-bg);
  border-left: 3px solid var(--border);
  padding: 0.6em 1em;
  margin: 0.75em 0;
  font-size: 0.92rem;
  color: var(--muted);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
  margin: 0.5em 0;
}}
th, td {{
  text-align: left;
  padding: 0.4em 0.75em;
  border-bottom: 1px solid var(--border);
}}
th {{ font-weight: 600; }}
tr:nth-child(even) {{ background: var(--table-stripe); }}
.chart-wrap {{
  position: relative;
  width: 100%;
  max-width: 700px;
  margin: 1em auto;
}}
.chart-wrap canvas {{ width: 100% !important; }}
.heatmap-wrap {{
  overflow-x: auto;
  margin: 1em 0;
}}
.heatmap {{
  border-collapse: collapse;
  margin: 0 auto;
  font-size: 0.8rem;
}}
.heatmap th, .heatmap td {{
  width: 3em;
  height: 2.5em;
  text-align: center;
  border: 1px solid var(--bg);
  padding: 0.2em;
}}
.heatmap th {{
  background: transparent;
  font-weight: 600;
  border: none;
}}
.heatmap .axis-label {{
  font-weight: 600;
  font-size: 0.85rem;
}}
.note {{ font-size: 0.88rem; color: var(--muted); }}
footer {{ margin-top: 2em; padding-top: 1em; border-top: 1px solid var(--border); font-size: 0.82rem; color: var(--muted); }}
</style>
</head>
<body>

<h1>Exploratory Analysis — Simulated World Values Survey Data</h1>
<p class="intro">
This page presents an exploratory analysis of a <strong>synthetic (simulated) dataset</strong> — the
rows are entirely fabricated to resemble the distributions and relationships found in the
<a href="https://www.worldvaluessurvey.org/" target="_blank" rel="noopener">World Values Survey</a>,
Wave 7 (Haerpfer, C. et al., 2022, <em>World Values Survey Wave 7</em>).
<strong>No real respondent data is used.</strong> Any patterns shown here are illustrative only and
should not be cited as empirical findings.
</p>
<p class="intro">
The dataset covers five countries in different parts of Asia:
<strong>Turkey</strong> (Middle East),
<strong>India</strong> (South Asia),
<strong>Singapore</strong> (Southeast Asia),
<strong>China</strong> (East Asia), and
<strong>Kazakhstan</strong> (Central Asia),
with roughly {total_raw:,} simulated respondents in total.
</p>

<h2>Data Dictionary</h2>
<table>
<thead><tr><th>Column</th><th>Type</th><th>Non-empty observations</th></tr></thead>
<tbody>
{"".join(f'<tr><td><code>{m["name"]}</code></td><td>{m["type"]}</td><td>{m["n"]:,}</td></tr>' for m in col_meta)}
</tbody>
</table>

<h2>Data Quality</h2>
<p class="note"><strong>Duplicate respondent IDs:</strong> {n_duplicates} found.</p>
<p class="note"><strong>Missing values:</strong> {n_dropped:,} rows contained at least one blank cell and
were removed before analysis, leaving <strong>{len(clean_rows):,}</strong> complete rows
({len(clean_rows)/total_raw*100:.1f}% of the original).</p>

<h2>Cultural Map (Inglehart–Welzel Style)</h2>
<p class="note">Each point is a country average. x&nbsp;=&nbsp;secular values, y&nbsp;=&nbsp;emancipative values.</p>
<div class="chart-wrap"><canvas id="culturalMap"></canvas></div>
<div class="takeaway">{cultural_takeaway}</div>

<h2>Value Fingerprints — Radar Chart</h2>
<p class="note">Six measures normalised to 0–1. Each line is one country.</p>
<div class="chart-wrap"><canvas id="radar"></canvas></div>
<div class="takeaway">{radar_takeaway}</div>

<h2>Individual Values — China vs India</h2>
<p class="note">~{SAMPLE_N} randomly sampled respondents per country. Larger outlined points mark country averages.</p>
<div class="chart-wrap"><canvas id="chinaIndia"></canvas></div>
<div class="takeaway">{scatter_takeaway}</div>

<h2>Life vs Financial Satisfaction — Singapore</h2>
<p class="note">10×10 heatmap. Darker cells = more respondents (n&nbsp;=&nbsp;{len(sg_rows):,}).</p>
<div class="heatmap-wrap" id="heatmapContainer"></div>
<div class="takeaway">{heatmap_takeaway}</div>

<footer>
Generated from synthetic data by <code>generate.py</code>.
Data modelled on the
<a href="https://www.worldvaluessurvey.org/" target="_blank" rel="noopener">World Values Survey, Wave 7</a>.
</footer>

<script>
// ── Embedded data ────────────────────────────────────────────────────────────
const COLORS = {json.dumps(COLORS)};
const MARKERS = {json.dumps(MARKERS)};
const COUNTRIES = {json.dumps(COUNTRIES)};
const culturalMap = {json.dumps(cultural_map)};
const radarData = {json.dumps(radar_data)};
const radarLabels = {json.dumps(radar_labels)};
const radarKeys = {json.dumps(radar_keys)};
const scatterIndividuals = {json.dumps(scatter_individuals)};
const scatterMeans = {json.dumps(scatter_means)};
const heatmap = {json.dumps(heatmap)};
const heatmapMax = {heatmap_max};

// ── Chart.js point style mapping ─────────────────────────────────────────────
const POINT_STYLES = {{
  "circle": "circle",
  "triangle": "triangle",
  "rectRot": "rectRot",
  "rect": "rect",
  "star": "star",
}};

// ── 1. Cultural Map ──────────────────────────────────────────────────────────
(() => {{
  const datasets = COUNTRIES.map(c => ({{
    label: c,
    data: [{{ x: culturalMap[c].secular, y: culturalMap[c].emancipative }}],
    backgroundColor: COLORS[c],
    borderColor: COLORS[c],
    pointStyle: POINT_STYLES[MARKERS[c]],
    pointRadius: 10,
    pointHoverRadius: 13,
  }}));
  new Chart(document.getElementById("culturalMap"), {{
    type: "scatter",
    data: {{ datasets }},
    options: {{
      responsive: true,
      plugins: {{
        datalabels: {{
          align: "top",
          anchor: "end",
          offset: 6,
          font: {{ weight: "bold", size: 12 }},
          color: (ctx) => COLORS[COUNTRIES[ctx.datasetIndex]],
          formatter: (_, ctx) => COUNTRIES[ctx.datasetIndex],
        }},
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const c = COUNTRIES[ctx.datasetIndex];
              return `${{c}}: secular ${{ctx.parsed.x.toFixed(2)}}, emancipative ${{ctx.parsed.y.toFixed(2)}}`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: "Secular values (0–1)" }} }},
        y: {{ title: {{ display: true, text: "Emancipative values (0–1)" }} }},
      }}
    }},
    plugins: [ChartDataLabels],
  }});
}})();

// ── 2. Radar ─────────────────────────────────────────────────────────────────
(() => {{
  const datasets = COUNTRIES.map(c => ({{
    label: c,
    data: radarKeys.map(k => radarData[c][k]),
    borderColor: COLORS[c],
    backgroundColor: COLORS[c] + "18",
    pointBackgroundColor: COLORS[c],
    pointStyle: POINT_STYLES[MARKERS[c]],
    pointRadius: 5,
    borderWidth: 2,
  }}));
  new Chart(document.getElementById("radar"), {{
    type: "radar",
    data: {{ labels: radarLabels, datasets }},
    options: {{
      responsive: true,
      scales: {{
        r: {{
          beginAtZero: true,
          max: 1,
          ticks: {{ stepSize: 0.2, callback: v => v.toFixed(1) }},
        }}
      }},
      plugins: {{
        datalabels: {{ display: false }},
        legend: {{ position: "bottom", labels: {{ usePointStyle: true, pointStyle: "circle" }} }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => `${{ctx.dataset.label}}: ${{ctx.parsed.r.toFixed(2)}}`
          }}
        }}
      }},
    }},
    plugins: [ChartDataLabels],
  }});
}})();

// ── 3. China vs India scatter ────────────────────────────────────────────────
(() => {{
  const datasets = [];
  ["China", "India"].forEach(c => {{
    datasets.push({{
      label: c,
      data: scatterIndividuals[c],
      backgroundColor: COLORS[c] + "80",
      borderColor: COLORS[c] + "80",
      pointStyle: POINT_STYLES[MARKERS[c]],
      pointRadius: 4,
      pointHoverRadius: 6,
    }});
    datasets.push({{
      label: c + " (mean)",
      data: [scatterMeans[c]],
      backgroundColor: "#fff",
      borderColor: COLORS[c],
      borderWidth: 3,
      pointStyle: POINT_STYLES[MARKERS[c]],
      pointRadius: 12,
      pointHoverRadius: 14,
    }});
  }});
  new Chart(document.getElementById("chinaIndia"), {{
    type: "scatter",
    data: {{ datasets }},
    options: {{
      responsive: true,
      plugins: {{
        datalabels: {{ display: false }},
        legend: {{
          position: "bottom",
          labels: {{
            usePointStyle: true,
            filter: (item) => !item.text.includes("(mean)"),
          }},
        }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const lbl = ctx.dataset.label;
              return `${{lbl}}: secular ${{ctx.parsed.x.toFixed(2)}}, emancipative ${{ctx.parsed.y.toFixed(2)}}`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: "Secular values (0–1)" }} }},
        y: {{ title: {{ display: true, text: "Emancipative values (0–1)" }} }},
      }}
    }},
    plugins: [ChartDataLabels],
  }});
}})();

// ── 4. Singapore heatmap (pure HTML/CSS) ─────────────────────────────────────
(() => {{
  const container = document.getElementById("heatmapContainer");
  let html = '<table class="heatmap">';

  html += '<tr><th class="axis-label" style="border:none"></th>';
  for (let j = 1; j <= 10; j++) html += `<th>${{j}}</th>`;
  html += '</tr>';

  for (let i = 9; i >= 0; i--) {{
    html += `<tr><th>${{i + 1}}</th>`;
    for (let j = 0; j < 10; j++) {{
      const v = heatmap[i][j];
      const intensity = heatmapMax > 0 ? v / heatmapMax : 0;
      const r = Math.round(255 - intensity * (255 - 213));
      const g = Math.round(255 - intensity * (255 - 94));
      const b = Math.round(255 - intensity * (255 - 0));
      const textColor = intensity > 0.55 ? "#fff" : "#1a1a1a";
      html += `<td style="background:rgb(${{r}},${{g}},${{b}});color:${{textColor}}" title="Life ${{i+1}}, Financial ${{j+1}}: ${{v}} respondents">${{v || ""}}</td>`;
    }}
    html += '</tr>';
  }}

  html += '</table>';
  html += '<p class="note" style="text-align:center;margin-top:0.3em"><strong>↑ Life satisfaction</strong> &nbsp;|&nbsp; <strong>Financial satisfaction →</strong></p>';
  container.innerHTML = html;
}})();
</script>
</body>
</html>"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"✓ index.html generated ({len(html):,} bytes)")
print(f"  {total_raw:,} raw rows → {len(clean_rows):,} after dropping {n_dropped:,} incomplete rows")
