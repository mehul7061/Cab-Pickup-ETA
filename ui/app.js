const select = (selector) => document.querySelector(selector);
const formatCount = new Intl.NumberFormat("en-IN");
const target = "pickup_eta_min";
const pageSize = 12;
const descriptions = {
  train: "train.csv · Raw training records with observed pick-up times",
  later: "later.csv · Later observations for monitoring and retraining",
  holdout_features: "holdout_features.csv · Features reserved for final evaluation",
};

let datasets;
let datasetName = "train";
let page = 0;

function element(tag, content) {
  const node = document.createElement(tag);
  node.textContent = content;
  return node;
}

function definitions(container, items) {
  container.replaceChildren(...items.map(([label, value]) => {
    const item = document.createElement("div");
    item.append(element("dt", label), element("dd", value));
    return item;
  }));
}

function zoneCounts(rows) {
  const counts = new Map();
  for (const row of rows) {
    counts.set(row.zone, (counts.get(row.zone) ?? 0) + 1);
  }
  return [...counts].sort(([left], [right]) => left.localeCompare(right));
}

function renderAuditEvidence(findings) {
  const { leaking_column, unit_mixup, sentinel_value, duplicate_rows,
    stuck_sensor, target_typos, spelling_variants } = findings;
  definitions(select("#evidence"), [
    ["Leaking column", leaking_column.column ?? "None detected"],
    ["Unit mix-up", unit_mixup.present
      ? `${unit_mixup.group_value} · ${unit_mixup.column} × ${unit_mixup.factor} as recorded`
      : "None detected"],
    ["Sentinel value", sentinel_value.present
      ? `${sentinel_value.column} = ${sentinel_value.value} · ${sentinel_value.count} rows`
      : "None detected"],
    ["Extra duplicate rows", formatCount.format(duplicate_rows.count ?? 0)],
    ["Stuck sensor", stuck_sensor.present
      ? `${stuck_sensor.column} · ${stuck_sensor.start_date} to ${stuck_sensor.end_date}`
      : "None detected"],
    ["Target typos", `${formatCount.format(target_typos.count ?? 0)} rows`],
    ["Spelling variants", spelling_variants.present
      ? `${spelling_variants.column} · ${spelling_variants.true_levels} clean categories`
      : "None detected"],
  ]);
}

function renderDataset() {
  const { columns, rows } = datasets[datasetName];
  const dates = rows.map(({ date }) => date).filter(Boolean).sort();
  const counts = zoneCounts(rows);
  select("#source").textContent = descriptions[datasetName];
  definitions(select("#metrics"), [
    ["Records", formatCount.format(rows.length)],
    ["Columns", formatCount.format(columns.length)],
    ["Date range", dates.length ? `${dates[0]} – ${dates.at(-1)}` : "No dates"],
    ["Observed ETA", columns.includes(target) ? "Available" : "Withheld"],
  ]);

  select("#zones").replaceChildren(...counts.map(([zone, count]) => {
    const item = document.createElement("li");
    const bar = document.createElement("meter");
    bar.max = rows.length;
    bar.value = count;
    bar.setAttribute("aria-label", zone);
    item.append(element("span", zone), bar, element("span", formatCount.format(count)));
    return item;
  }));

  const allZones = element("option", "All zones");
  allZones.value = "";
  select("#zone-filter").replaceChildren(allZones, ...counts.map(([zone]) => element("option", zone)));
  const heading = document.createElement("tr");
  heading.append(...columns.map((column) => {
    const cell = element("th", column);
    cell.scope = "col";
    return cell;
  }));
  select("#dashboard thead").replaceChildren(heading);
  renderRows();
}

function renderRows() {
  const { columns, rows } = datasets[datasetName];
  const zone = select("#zone-filter").value;
  const filtered = zone ? rows.filter((row) => row.zone === zone) : rows;
  const start = page * pageSize;
  const end = Math.min(start + pageSize, filtered.length);

  select("#dashboard tbody").replaceChildren(...filtered.slice(start, end).map((row) => {
    const record = document.createElement("tr");
    record.append(...columns.map((column) => element("td", row[column] || "—")));
    return record;
  }));
  select("#page-status").textContent = filtered.length
    ? `${start + 1}–${end} of ${formatCount.format(filtered.length)} records`
    : "No records";
  select("#previous").disabled = page === 0;
  select("#next").disabled = end >= filtered.length;
  select(".table-scroll").scrollLeft = 0;
}

async function load() {
  try {
    const response = await fetch("/api/datasets");
    if (!response.ok) throw new Error("Dataset request failed");
    const result = await response.json();
    datasets = result.datasets;
    renderAuditEvidence(result.findings);
    renderDataset();
    select("#dashboard").hidden = false;
    select("#loading").hidden = true;
  } catch {
    select("#loading").textContent = "Could not load the data. Run python dashboard.py from the extracted project folder and reload.";
  }
}

const datasetButtons = document.querySelectorAll("[data-dataset]");
for (const button of datasetButtons) {
  button.addEventListener("click", () => {
    datasetName = button.dataset.dataset;
    page = 0;
    for (const choice of datasetButtons) {
      choice.setAttribute("aria-pressed", choice === button);
    }
    renderDataset();
  });
}

select("#zone-filter").addEventListener("change", () => {
  page = 0;
  renderRows();
});
select("#previous").addEventListener("click", () => {
  page -= 1;
  renderRows();
});
select("#next").addEventListener("click", () => {
  page += 1;
  renderRows();
});

load();

async function loadTraining() {
  const status = select("#training-status");
  try {
    const response = await fetch("/api/training");
    if (!response.ok) throw new Error("Training results unavailable");
    const runs = await response.json();
    if (!runs.length) {
      status.textContent = "No completed training runs found.";
      return;
    }
    select("#training-runs").replaceChildren(...runs.map((run) => {
      const row = document.createElement("tr");
      const model = run.champion_model
        ? `${run.champion_model} v${run.champion_version} (champion)`
        : "—";
      row.append(element("td", run.name), element("td", run.mae.toFixed(3)), element("td", model));
      return row;
    }));
    status.textContent = "Lower MAE means the predicted pick-up time was closer to the actual time.";
    select("#training-table").hidden = false;
  } catch {
    status.textContent = "Training results unavailable. Check that mlflow.db is in the project folder.";
  }
}

loadTraining();

select("#prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const output = select("#prediction-result");
  const values = Object.fromEntries(new FormData(form));
  for (const input of form.querySelectorAll("input[type=number]")) {
    values[input.name] = input.valueAsNumber;
  }
  output.textContent = "Predicting…";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    if (!response.ok) {
      output.textContent = response.status === 422
        ? "Check the booking values."
        : "Start the Stage 3 API to make predictions.";
      return;
    }
    const result = await response.json();
    output.textContent = `Estimated pick-up time: ${result.pickup_eta_min.toFixed(1)} minutes`;
  } catch {
    output.textContent = "Could not reach the prediction API.";
  }
});
