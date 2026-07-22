"""Static HTML prediction dashboard for the final AeroSurrogate model."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from aero_surrogate.data import FEATURE_COLUMNS, TARGET_COLUMNS, load_dataset


def export_html_dashboard(
    dataset_path: str | Path,
    run_dir: str | Path,
    output_path: str | Path = "output/dashboard/aerosurrogate_dashboard.html",
    model_path: str | Path | None = None,
) -> Path:
    """Export a self-contained browser interface for coefficient prediction.

    ``run_dir`` identifies the reproducible validation run. ``model_path`` may
    point to the deployment model trained on the complete flow5 dataset. When
    omitted, the model saved inside the reproducible run is used.
    """

    dataset_path = Path(dataset_path)
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    deployment_model = (
        Path(model_path)
        if model_path is not None
        else run_dir / "outputs" / "deployment_model.pkl"
    )

    dataset = _prepare_dataset(load_dataset(dataset_path))
    summary_path = run_dir / "reports" / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    run_id = str(summary.get("run_id", run_dir.name))
    reynolds_levels = sorted(float(value) for value in dataset["reynolds"].unique())

    payload = {
        "dataset": _records(
            dataset,
            ["naca", "alpha_deg", "reynolds", *TARGET_COLUMNS],
        ),
        "airfoils": sorted(dataset["naca"].unique().tolist()),
        "reynolds_levels": reynolds_levels,
        "alpha_range": [
            float(dataset["alpha_deg"].min()),
            float(dataset["alpha_deg"].max()),
        ],
        "feature_ranges": {
            feature: [float(dataset[feature].min()), float(dataset[feature].max())]
            for feature in FEATURE_COLUMNS
        },
        "surrogate": _serialize_surrogate(deployment_model),
    }

    html = (
        _html_template()
        .replace(
            "__DASHBOARD_PAYLOAD__",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        .replace("__RUN_ID__", run_id)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    prepared = dataset.copy()
    return prepared.sort_values(["naca", "reynolds", "alpha_deg"]).reset_index(drop=True)


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    exported = frame.loc[:, columns].copy()
    for column in exported.columns:
        if pd.api.types.is_numeric_dtype(exported[column]):
            exported[column] = exported[column].astype(float).round(8)
    return exported.to_dict(orient="records")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _serialize_surrogate(path: Path) -> dict[str, Any]:
    """Serialize a fitted Random Forest for prediction in a static dashboard."""

    with path.open("rb") as file:
        surrogate = pickle.load(file)

    forest = getattr(surrogate, "model", None)
    estimators = getattr(forest, "estimators_", None)
    if not estimators:
        return {"available": False, "model_type": type(surrogate).__name__}

    trees = []
    for estimator in estimators:
        tree = estimator.tree_
        trees.append(
            {
                "left": tree.children_left.astype(int).tolist(),
                "right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.round(12).tolist(),
                "value": tree.value[:, :, 0].round(12).tolist(),
            }
        )

    return {
        "available": True,
        "model_type": "random-forest",
        "features": list(surrogate.feature_columns),
        "targets": list(surrogate.target_columns),
        "trees": trees,
    }


def _html_template() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AeroSurrogate</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f2f4f1;
      --panel: #ffffff;
      --ink: #16201b;
      --muted: #637069;
      --line: #d9dfda;
      --soft: #eef2ee;
      --green: #176b55;
      --blue: #315f9e;
      --red: #a34f43;
      --amber: #9a681c;
      --shadow: 0 10px 30px rgba(22, 32, 27, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select { font: inherit; }
    .app-shell {
      width: min(1440px, 100%);
      min-height: 100vh;
      margin: 0 auto;
      padding: 22px;
    }
    header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      padding: 2px 2px 18px;
    }
    h1 { margin: 0; font-size: 28px; font-weight: 650; letter-spacing: 0; }
    .tagline { margin: 5px 0 0; color: var(--muted); font-size: 14px; }
    .status-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      margin-right: 7px;
      border-radius: 50%;
      background: var(--green);
    }
    .model-status { color: var(--muted); font-size: 13px; white-space: nowrap; }
    .workspace {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      min-height: 710px;
      overflow: hidden;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .controls {
      padding: 22px;
      background: #fbfcfa;
      border-right: 1px solid var(--line);
    }
    .controls h2, .visual h2 { margin: 0; font-size: 17px; font-weight: 650; }
    .section-copy { margin: 6px 0 20px; color: var(--muted); font-size: 13px; line-height: 1.4; }
    .field { display: grid; gap: 7px; margin-bottom: 16px; }
    .field-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    label, .field-label { color: var(--muted); font-size: 12px; font-weight: 600; }
    .unit { color: #8a948e; font-size: 11px; }
    select, input[type="number"] {
      width: 100%;
      min-height: 42px;
      padding: 8px 10px;
      color: var(--ink);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    input[type="range"] { width: 100%; accent-color: var(--green); }
    .alpha-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 76px;
      gap: 10px;
      align-items: center;
    }
    .geometry-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .geometry-grid .field:last-child { grid-column: 1 / -1; }
    button {
      width: 100%;
      min-height: 44px;
      margin-top: 2px;
      border: 0;
      border-radius: 6px;
      color: #fff;
      background: var(--green);
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: #105641; }
    button:focus-visible, select:focus-visible, input:focus-visible {
      outline: 3px solid rgba(49, 95, 158, 0.24);
      outline-offset: 2px;
    }
    .domain-note {
      min-height: 52px;
      margin: 14px 0 0;
      padding: 10px 11px;
      border-left: 3px solid var(--green);
      background: var(--soft);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .domain-note.warning { border-left-color: var(--amber); background: #fbf5e9; color: #6d501e; }
    .visual { min-width: 0; padding: 22px 26px 18px; }
    .visual-header {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 14px;
    }
    .visual-title p { margin: 5px 0 0; color: var(--muted); font-size: 13px; }
    .coefficient-values {
      display: grid;
      grid-template-columns: repeat(3, minmax(92px, 1fr));
      gap: 8px;
    }
    .value-box {
      min-width: 92px;
      padding: 9px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    .value-box span { display: block; color: var(--muted); font-size: 11px; font-weight: 700; }
    .value-box strong { display: block; margin-top: 2px; font-size: 21px; font-variant-numeric: tabular-nums; }
    .value-box.cl { border-top: 3px solid var(--green); }
    .value-box.cd { border-top: 3px solid var(--red); }
    .value-box.cm { border-top: 3px solid var(--blue); }
    .chart-wrap { width: 100%; min-height: 560px; }
    svg { display: block; width: 100%; height: auto; }
    .grid-line { stroke: #e5e9e5; stroke-width: 1; }
    .axis-line { stroke: #aeb8b1; stroke-width: 1; }
    .axis-label, .tick { fill: var(--muted); font-size: 11px; }
    .facet-label { fill: var(--ink); font-size: 13px; font-weight: 700; }
    .model-line { fill: none; stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }
    .flow5-point { fill: #fff; stroke-width: 1.7; }
    .alpha-marker { stroke: #27332d; stroke-width: 1; stroke-dasharray: 4 4; opacity: 0.7; }
    .selected-point { stroke: #fff; stroke-width: 2.5; }
    .legend { display: flex; gap: 18px; margin-top: 4px; color: var(--muted); font-size: 12px; }
    .legend-line { display: inline-block; width: 24px; height: 3px; margin-right: 6px; vertical-align: middle; background: var(--green); }
    .legend-dot { display: inline-block; width: 9px; height: 9px; margin-right: 7px; vertical-align: middle; border: 2px solid var(--green); border-radius: 50%; background: #fff; }
    @media (max-width: 900px) {
      .app-shell { padding: 14px; }
      .workspace { grid-template-columns: 1fr; }
      .controls { border-right: 0; border-bottom: 1px solid var(--line); }
      .visual-header { align-items: stretch; flex-direction: column; }
      .chart-wrap { min-height: 430px; }
    }
    @media (max-width: 520px) {
      header { align-items: start; flex-direction: column; }
      .model-status { white-space: normal; }
      .controls, .visual { padding: 18px; }
      .geometry-grid { grid-template-columns: 1fr; gap: 0; }
      .geometry-grid .field:last-child { grid-column: auto; }
      .coefficient-values { grid-template-columns: repeat(3, 1fr); }
      .value-box { min-width: 0; padding: 8px; }
      .value-box strong { font-size: 16px; }
    }
  </style>
</head>
<body>
<main class="app-shell">
  <header>
    <div>
      <h1>AeroSurrogate</h1>
      <p class="tagline">Interactive airfoil coefficient predictor</p>
    </div>
    <div class="model-status"><span class="status-dot"></span>Model ready</div>
  </header>

  <section class="workspace">
    <aside id="control-panel" class="controls">
      <h2>Airfoil inputs</h2>
      <p class="section-copy">Change the geometry and operating point to update the prediction.</p>
      <form id="prediction-form">
        <label class="field">
          <span class="field-head"><span>NACA airfoil</span><span class="unit">preset</span></span>
          <select id="predict-naca"></select>
        </label>

        <div class="geometry-grid">
          <label class="field">
            <span class="field-head"><span>Camber</span><span class="unit">% chord</span></span>
            <input id="predict-camber" type="number" min="0" max="9" step="0.1" value="2">
          </label>
          <label class="field">
            <span class="field-head"><span>Camber position</span><span class="unit">% chord</span></span>
            <input id="predict-position" type="number" min="0" max="90" step="1" value="40">
          </label>
          <label class="field">
            <span class="field-head"><span>Thickness</span><span class="unit">% chord</span></span>
            <input id="predict-thickness" type="number" min="1" max="40" step="0.1" value="12">
          </label>
        </div>

        <div class="field">
          <span class="field-head"><label for="predict-alpha-range">Angle of attack</label><span class="unit">degrees</span></span>
          <div class="alpha-row">
            <input id="predict-alpha-range" type="range" min="-6" max="14" step="0.1" value="4">
            <input id="predict-alpha" type="number" min="-6" max="14" step="0.1" value="4" aria-label="Angle of attack in degrees">
          </div>
        </div>

        <label class="field">
          <span class="field-head"><span>Reynolds number</span><span class="unit">Re</span></span>
          <input id="predict-reynolds" type="number" min="500000" max="2000000" step="10000" value="1000000" list="reynolds-options">
          <datalist id="reynolds-options"></datalist>
        </label>

        <button id="predict-button" type="submit">Predict coefficients</button>
      </form>
      <p id="prediction-domain" class="domain-note" aria-live="polite"></p>
    </aside>

    <section class="visual">
      <div class="visual-header">
        <div class="visual-title">
          <h2>Predicted aerodynamic coefficients</h2>
          <p id="selection-label">NACA 2412 at Re 1,000,000</p>
        </div>
        <div class="coefficient-values" aria-live="polite">
          <div class="value-box cl"><span>CL</span><strong id="prediction-cl">--</strong></div>
          <div class="value-box cd"><span>CD</span><strong id="prediction-cd">--</strong></div>
          <div class="value-box cm"><span>CM</span><strong id="prediction-cm">--</strong></div>
        </div>
      </div>
      <div class="chart-wrap">
        <svg id="coefficient-chart" viewBox="0 0 940 590" role="img" aria-label="CL, CD and CM versus angle of attack"></svg>
      </div>
      <div class="legend">
        <span><i class="legend-line"></i>Random Forest prediction</span>
        <span><i class="legend-dot"></i>flow5 data for selected preset</span>
      </div>
    </section>
  </section>
</main>

<script>
const payload = __DASHBOARD_PAYLOAD__;
const colors = { cl: "#176b55", cd: "#a34f43", cm: "#315f9e" };
const labels = { cl: "CL", cd: "CD", cm: "CM" };
const chart = document.getElementById("coefficient-chart");
const form = document.getElementById("prediction-form");
const nacaInput = document.getElementById("predict-naca");
const camberInput = document.getElementById("predict-camber");
const positionInput = document.getElementById("predict-position");
const thicknessInput = document.getElementById("predict-thickness");
const alphaRange = document.getElementById("predict-alpha-range");
const alphaInput = document.getElementById("predict-alpha");
const reynoldsInput = document.getElementById("predict-reynolds");
const reynoldsOptions = document.getElementById("reynolds-options");
const domainNote = document.getElementById("prediction-domain");
const selectionLabel = document.getElementById("selection-label");
const outputs = {
  cl: document.getElementById("prediction-cl"),
  cd: document.getElementById("prediction-cd"),
  cm: document.getElementById("prediction-cm")
};

function fmt(value, digits = 4) {
  if (!Number.isFinite(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function nacaGeometry(name) {
  const code = name.replace(/\D/g, "").slice(-4);
  if (code.length !== 4) return null;
  const camber = Number(code[0]);
  return {
    camber,
    position: camber === 0 ? 0 : Number(code[1]) * 10,
    thickness: Number(code.slice(2))
  };
}

function syncGeometryFromPreset() {
  if (nacaInput.value === "manual") return;
  const geometry = nacaGeometry(nacaInput.value);
  if (!geometry) return;
  camberInput.value = geometry.camber;
  positionInput.value = geometry.position;
  thicknessInput.value = geometry.thickness;
}

function currentGeometry() {
  return {
    camber: Number(camberInput.value) / 100,
    camber_position: Number(positionInput.value) / 100,
    thickness: Number(thicknessInput.value) / 100
  };
}

function featureVector(alpha) {
  const geometry = currentGeometry();
  const values = {
    ...geometry,
    alpha_deg: Number(alpha),
    reynolds: Number(reynoldsInput.value)
  };
  return payload.surrogate.features.map(feature => values[feature]);
}

function predictForest(input) {
  const forest = payload.surrogate;
  const total = new Array(forest.targets.length).fill(0);
  for (const tree of forest.trees) {
    let node = 0;
    while (tree.feature[node] >= 0) {
      const feature = tree.feature[node];
      node = input[feature] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    }
    tree.value[node].forEach((value, index) => { total[index] += value; });
  }
  return total.map(value => value / forest.trees.length);
}

function predictionAt(alpha) {
  const prediction = predictForest(featureVector(alpha));
  return Object.fromEntries(payload.surrogate.targets.map((target, index) => [target, prediction[index]]));
}

function seriesPredictions() {
  const [minimum, maximum] = payload.alpha_range;
  const rows = [];
  for (let alpha = minimum; alpha <= maximum + 1e-9; alpha += 0.25) {
    rows.push({ alpha_deg: Number(alpha.toFixed(2)), ...predictionAt(alpha) });
  }
  return rows;
}

function actualRows() {
  if (nacaInput.value === "manual") return [];
  const reynolds = Number(reynoldsInput.value);
  return payload.dataset
    .filter(row => row.naca === nacaInput.value && row.reynolds === reynolds)
    .sort((a, b) => a.alpha_deg - b.alpha_deg);
}

function extent(values) {
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) { minimum -= 1; maximum += 1; }
  const padding = (maximum - minimum) * 0.1;
  return [minimum - padding, maximum + padding];
}

function scale(value, domain, range) {
  return range[0] + ((value - domain[0]) / (domain[1] - domain[0])) * (range[1] - range[0]);
}

function pathFor(rows, target, xDomain, yDomain, box) {
  return rows.map((row, index) => {
    const x = scale(row.alpha_deg, xDomain, [box.left, box.right]);
    const y = scale(row[target], yDomain, [box.bottom, box.top]);
    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
}

function renderChart(predicted, actual) {
  const width = 940;
  const left = 68;
  const right = width - 22;
  const facetHeight = 154;
  const facetGap = 24;
  const firstTop = 26;
  const xDomain = payload.alpha_range;
  const selectedAlpha = Number(alphaInput.value);
  let html = "";

  ["cl", "cd", "cm"].forEach((target, index) => {
    const top = firstTop + index * (facetHeight + facetGap);
    const bottom = top + facetHeight;
    const box = { left, right, top, bottom };
    const values = [...predicted.map(row => row[target]), ...actual.map(row => row[target])];
    const yDomain = extent(values);

    for (let tick = 0; tick <= 4; tick++) {
      const yValue = yDomain[0] + (yDomain[1] - yDomain[0]) * tick / 4;
      const y = scale(yValue, yDomain, [bottom, top]);
      html += `<line class="grid-line" x1="${left}" y1="${y}" x2="${right}" y2="${y}"></line>`;
      html += `<text class="tick" x="${left - 10}" y="${y + 4}" text-anchor="end">${fmt(yValue, target === "cd" ? 3 : 2)}</text>`;
    }
    html += `<line class="axis-line" x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}"></line>`;
    html += `<text class="facet-label" x="${left}" y="${top - 9}">${labels[target]}</text>`;
    html += `<path class="model-line" stroke="${colors[target]}" d="${pathFor(predicted, target, xDomain, yDomain, box)}"></path>`;

    actual.forEach(row => {
      const x = scale(row.alpha_deg, xDomain, [left, right]);
      const y = scale(row[target], yDomain, [bottom, top]);
      html += `<circle class="flow5-point" stroke="${colors[target]}" cx="${x}" cy="${y}" r="3.3"><title>flow5: alpha ${row.alpha_deg}, ${labels[target]} ${fmt(row[target], 5)}</title></circle>`;
    });

    const markerX = scale(selectedAlpha, xDomain, [left, right]);
    const selected = predictionAt(selectedAlpha)[target];
    const markerY = scale(selected, yDomain, [bottom, top]);
    html += `<line class="alpha-marker" x1="${markerX}" y1="${top}" x2="${markerX}" y2="${bottom}"></line>`;
    html += `<circle class="selected-point" fill="${colors[target]}" cx="${markerX}" cy="${markerY}" r="6"><title>Selected alpha ${selectedAlpha}: ${labels[target]} ${fmt(selected, 5)}</title></circle>`;

    if (index === 2) {
      for (let tick = 0; tick <= 5; tick++) {
        const xValue = xDomain[0] + (xDomain[1] - xDomain[0]) * tick / 5;
        const x = scale(xValue, xDomain, [left, right]);
        html += `<line class="axis-line" x1="${x}" y1="${bottom}" x2="${x}" y2="${bottom + 5}"></line>`;
        html += `<text class="tick" x="${x}" y="${bottom + 19}" text-anchor="middle">${fmt(xValue, 0)}</text>`;
      }
      html += `<text class="axis-label" x="${(left + right) / 2}" y="${bottom + 39}" text-anchor="middle">Angle of attack [deg]</text>`;
    }
  });
  chart.innerHTML = html;
}

function updateDomainStatus() {
  const values = featureVector(Number(alphaInput.value));
  const outside = payload.surrogate.features.filter((feature, index) => {
    const [minimum, maximum] = payload.feature_ranges[feature];
    return values[index] < minimum || values[index] > maximum;
  });
  const reynolds = Number(reynoldsInput.value);
  if (outside.length) {
    domainNote.className = "domain-note warning";
    domainNote.textContent = `Outside trained range: ${outside.join(", ")}. Treat this estimate with caution.`;
  } else if (!payload.reynolds_levels.includes(reynolds)) {
    domainNote.className = "domain-note warning";
    domainNote.textContent = "Reynolds is between sampled flow5 levels. The Random Forest estimate is piecewise between those levels.";
  } else {
    domainNote.className = "domain-note";
    domainNote.textContent = "Inside the trained flow5 parameter range.";
  }
}

function updatePrediction() {
  if (!payload.surrogate.available) {
    domainNote.className = "domain-note warning";
    domainNote.textContent = "The embedded model is not a fitted Random Forest.";
    return;
  }
  const numericInputs = [camberInput, positionInput, thicknessInput, alphaInput, reynoldsInput]
    .map(input => Number(input.value));
  if (numericInputs.some(value => !Number.isFinite(value))) {
    domainNote.className = "domain-note warning";
    domainNote.textContent = "All inputs must be numeric.";
    return;
  }

  const current = predictionAt(Number(alphaInput.value));
  outputs.cl.textContent = fmt(current.cl, 5);
  outputs.cd.textContent = fmt(current.cd, 6);
  outputs.cm.textContent = fmt(current.cm, 5);
  const name = nacaInput.value === "manual" ? "Manual geometry" : nacaInput.value;
  selectionLabel.textContent = `${name} at Re ${Number(reynoldsInput.value).toLocaleString()} and alpha ${fmt(alphaInput.value, 1)} deg`;
  updateDomainStatus();
  renderChart(seriesPredictions(), actualRows());
}

function renderControls() {
  nacaInput.innerHTML = payload.airfoils.map(name => `<option value="${name}">${name}</option>`).join("")
    + '<option value="manual">Manual geometry</option>';
  nacaInput.value = payload.airfoils.includes("NACA 2412") ? "NACA 2412" : payload.airfoils[0];
  reynoldsOptions.innerHTML = payload.reynolds_levels
    .map(value => `<option value="${value}">${value.toLocaleString()}</option>`)
    .join("");
  reynoldsInput.min = payload.feature_ranges.reynolds[0];
  reynoldsInput.max = payload.feature_ranges.reynolds[1];
  alphaRange.min = payload.alpha_range[0];
  alphaRange.max = payload.alpha_range[1];
  alphaInput.min = payload.alpha_range[0];
  alphaInput.max = payload.alpha_range[1];
  syncGeometryFromPreset();
}

form.addEventListener("submit", event => { event.preventDefault(); updatePrediction(); });
nacaInput.addEventListener("change", () => { syncGeometryFromPreset(); updatePrediction(); });
[camberInput, positionInput, thicknessInput].forEach(input => {
  input.addEventListener("input", () => { nacaInput.value = "manual"; updatePrediction(); });
});
alphaRange.addEventListener("input", () => { alphaInput.value = alphaRange.value; updatePrediction(); });
alphaInput.addEventListener("input", () => { alphaRange.value = alphaInput.value; updatePrediction(); });
reynoldsInput.addEventListener("input", updatePrediction);

renderControls();
updatePrediction();
</script>
</body>
</html>
"""
