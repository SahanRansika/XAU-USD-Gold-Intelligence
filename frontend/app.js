/**
 * Frontend client logic for XAU/USD Gold Price Prediction ML Dashboard
 */

let priceChart = null;
let indicatorChart = null;
let currentActiveIndicator = "rsi"; // 'rsi' or 'macd'
let cachedCandles = [];

const API_BASE = window.location.origin;

document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  loadDashboardData();
  loadBenchmarkMetrics();
});

function initEventListeners() {
  // Model selector change
  document.getElementById("model-select").addEventListener("change", (e) => {
    document.getElementById("ticker-active-model").innerText = e.target.options[e.target.selectedIndex].text;
    fetchLatestPrediction();
  });

  // Run prediction button
  document.getElementById("btn-predict-now").addEventListener("click", fetchLatestPrediction);

  // Refresh data button
  document.getElementById("btn-refresh-data").addEventListener("click", () => {
    loadDashboardData();
    fetchLatestPrediction();
  });

  // Candle count selector
  document.getElementById("candle-count-select").addEventListener("change", () => {
    loadDashboardData();
  });

  // Indicator toggles on main chart
  ["toggle-sma", "toggle-ema", "toggle-bb"].forEach(id => {
    document.getElementById(id).addEventListener("change", updateChartDatasets);
  });

  // Indicator sub-tabs (RSI vs MACD)
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      e.target.classList.add("active");
      currentActiveIndicator = e.target.getAttribute("data-target");
      renderIndicatorChart(cachedCandles);
    });
  });

  // Simulator Form
  document.getElementById("simulator-form").addEventListener("submit", handleScenarioSimulation);

  // Twelve Data Modal
  const modal = document.getElementById("api-modal");
  document.getElementById("btn-open-api-modal").addEventListener("click", () => modal.classList.remove("hidden"));
  document.getElementById("btn-close-modal").addEventListener("click", () => modal.classList.add("hidden"));
  document.getElementById("btn-cancel-modal").addEventListener("click", () => modal.classList.add("hidden"));
  document.getElementById("btn-save-fetch-api").addEventListener("click", handleTwelveDataFetch);
}

/** Load historical candle and indicator data */
async function loadDashboardData() {
  const limit = document.getElementById("candle-count-select").value || 100;
  try {
    const res = await fetch(`${API_BASE}/api/historical?limit=${limit}`);
    if (!res.ok) throw new Error("Failed to fetch historical data");
    const data = await res.json();
    cachedCandles = data.candles || [];

    if (cachedCandles.length > 0) {
      const latest = cachedCandles[cachedCandles.length - 1];
      updateTicker(latest);
      populateSimulatorDefaults(latest);
      renderPriceChart(cachedCandles);
      renderIndicatorChart(cachedCandles);
    }
  } catch (err) {
    console.error("Dashboard data load error:", err);
  }
}

/** Update the top ticker metrics */
function updateTicker(latest) {
  document.getElementById("ticker-current-price").innerText = `$${latest.close.toFixed(2)}`;
  
  // Calculate 24h high/low from last 24 candles
  const last24 = cachedCandles.slice(-24);
  const high24 = Math.max(...last24.map(c => c.high));
  const low24 = Math.min(...last24.map(c => c.low));
  document.getElementById("ticker-high-low").innerText = `$${high24.toFixed(2)} / $${low24.toFixed(2)}`;
}

/** Populate simulator form with recent market state */
function populateSimulatorDefaults(latest) {
  document.getElementById("sim-open").value = latest.open.toFixed(2);
  document.getElementById("sim-high").value = latest.high.toFixed(2);
  document.getElementById("sim-low").value = latest.low.toFixed(2);
  document.getElementById("sim-close").value = latest.close.toFixed(2);
}

/** Render Main Price Chart with Overlays */
function renderPriceChart(candles) {
  const ctx = document.getElementById("priceChart").getContext("2d");
  const labels = candles.map(c => c.datetime.split(" ")[1] || c.datetime);
  const closePrices = candles.map(c => c.close);
  const sma20 = candles.map(c => c.sma_20);
  const ema50 = candles.map(c => c.ema_50);
  const bbUpper = candles.map(c => c.bb_upper);
  const bbLower = candles.map(c => c.bb_lower);

  const showSma = document.getElementById("toggle-sma").checked;
  const showEma = document.getElementById("toggle-ema").checked;
  const showBb = document.getElementById("toggle-bb").checked;

  const datasets = [
    {
      label: "Gold Close ($)",
      data: closePrices,
      borderColor: "#d4af37",
      backgroundColor: "rgba(212, 175, 55, 0.08)",
      borderWidth: 2,
      fill: true,
      tension: 0.1,
      pointRadius: 0,
      pointHoverRadius: 5
    },
    {
      label: "SMA (20)",
      data: sma20,
      borderColor: "#00b4d8",
      borderWidth: 1.5,
      borderDash: [4, 4],
      fill: false,
      pointRadius: 0,
      hidden: !showSma
    },
    {
      label: "EMA (50)",
      data: ema50,
      borderColor: "#9d4edd",
      borderWidth: 1.5,
      fill: false,
      pointRadius: 0,
      hidden: !showEma
    },
    {
      label: "BB Upper",
      data: bbUpper,
      borderColor: "rgba(255, 255, 255, 0.25)",
      borderWidth: 1,
      fill: false,
      pointRadius: 0,
      hidden: !showBb
    },
    {
      label: "BB Lower",
      data: bbLower,
      borderColor: "rgba(255, 255, 255, 0.25)",
      borderWidth: 1,
      fill: "-1",
      backgroundColor: "rgba(255, 255, 255, 0.02)",
      pointRadius: 0,
      hidden: !showBb
    }
  ];

  if (priceChart) {
    priceChart.data.labels = labels;
    priceChart.data.datasets = datasets;
    priceChart.update();
  } else {
    priceChart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#141822",
            borderColor: "#232a3b",
            borderWidth: 1,
            titleFont: { family: "Inter", weight: "600" },
            bodyFont: { family: "JetBrains Mono" }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } }
          },
          y: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: {
              color: "#94a3b8",
              font: { family: "JetBrains Mono", size: 11 },
              callback: (val) => `$${val.toFixed(0)}`
            }
          }
        }
      }
    });
  }
}

function updateChartDatasets() {
  if (!priceChart) return;
  const showSma = document.getElementById("toggle-sma").checked;
  const showEma = document.getElementById("toggle-ema").checked;
  const showBb = document.getElementById("toggle-bb").checked;

  priceChart.data.datasets[1].hidden = !showSma;
  priceChart.data.datasets[2].hidden = !showEma;
  priceChart.data.datasets[3].hidden = !showBb;
  priceChart.data.datasets[4].hidden = !showBb;
  priceChart.update();
}

/** Render Secondary Chart (RSI or MACD) */
function renderIndicatorChart(candles) {
  const ctx = document.getElementById("indicatorChart").getContext("2d");
  const labels = candles.map(c => c.datetime.split(" ")[1] || c.datetime);
  const latest = candles[candles.length - 1];

  let datasets = [];
  let yAxisConfig = {};

  if (currentActiveIndicator === "rsi") {
    const rsiValues = candles.map(c => c.rsi_14);
    document.getElementById("indicator-val").innerText = `RSI: ${latest.rsi_14.toFixed(2)}`;

    datasets = [
      {
        label: "RSI (14)",
        data: rsiValues,
        borderColor: "#f59e0b",
        borderWidth: 1.8,
        pointRadius: 0,
        fill: false
      }
    ];

    yAxisConfig = {
      min: 10,
      max: 90,
      grid: { color: "rgba(255, 255, 255, 0.05)" },
      ticks: { color: "#64748b", stepSize: 20 }
    };
  } else {
    const macdValues = candles.map(c => c.macd);
    const macdSignals = candles.map(c => c.macd_signal);
    document.getElementById("indicator-val").innerText = `MACD: ${latest.macd.toFixed(3)}`;

    datasets = [
      {
        label: "MACD",
        data: macdValues,
        borderColor: "#06b6d4",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false
      },
      {
        label: "Signal",
        data: macdSignals,
        borderColor: "#ec4899",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false
      }
    ];

    yAxisConfig = {
      grid: { color: "rgba(255, 255, 255, 0.05)" },
      ticks: { color: "#64748b" }
    };
  }

  if (indicatorChart) {
    indicatorChart.data.labels = labels;
    indicatorChart.data.datasets = datasets;
    indicatorChart.options.scales.y = yAxisConfig;
    indicatorChart.update();
  } else {
    indicatorChart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: yAxisConfig
        }
      }
    });
  }
}

/** Fetch and display prediction for latest market state */
async function fetchLatestPrediction() {
  const model = document.getElementById("model-select").value;
  try {
    const res = await fetch(`${API_BASE}/api/predict/latest?model=${model}`);
    if (!res.ok) throw new Error("Prediction request failed");
    const data = await res.json();

    document.getElementById("pred-price").innerText = `$${data.predicted_price.toFixed(2)}`;
    const changeSign = data.predicted_change >= 0 ? "+" : "";
    const changeColor = data.predicted_change >= 0 ? "var(--bullish)" : "var(--bearish)";
    
    const changeEl = document.getElementById("pred-change");
    changeEl.innerText = `${changeSign}$${data.predicted_change.toFixed(2)} (${changeSign}${data.predicted_pct_change.toFixed(2)}%)`;
    changeEl.style.color = changeColor;

    const signalBadge = document.getElementById("signal-badge");
    signalBadge.className = `signal-badge ${data.signal.toLowerCase()}`;
    signalBadge.innerText = `${data.signal} ${data.signal === "BULLISH" ? "▲" : "▼"}`;

    document.getElementById("metric-rsi").innerText = data.rsi_14.toFixed(1);
    document.getElementById("metric-macd").innerText = data.macd.toFixed(2);
    document.getElementById("metric-sma").innerText = `$${data.sma_20.toFixed(1)}`;
    document.getElementById("metric-ema").innerText = `$${data.ema_50.toFixed(1)}`;

  } catch (err) {
    console.error("Inference fetch error:", err);
  }
}

/** Simulate prediction from custom OHLC candle */
async function handleScenarioSimulation(e) {
  e.preventDefault();
  const payload = {
    open: parseFloat(document.getElementById("sim-open").value),
    high: parseFloat(document.getElementById("sim-high").value),
    low: parseFloat(document.getElementById("sim-low").value),
    close: parseFloat(document.getElementById("sim-close").value),
    volume: parseFloat(document.getElementById("sim-volume").value),
    model_choice: document.getElementById("model-select").value
  };

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Simulation failed");
    const result = await res.json();

    const simResultBox = document.getElementById("sim-result");
    simResultBox.classList.remove("hidden");
    document.getElementById("sim-res-price").innerText = `$${result.predicted_price.toFixed(2)} (${result.predicted_change >= 0 ? "+" : ""}${result.predicted_change.toFixed(2)})`;
    
    const dirEl = document.getElementById("sim-res-dir");
    dirEl.innerText = result.signal;
    dirEl.className = `badge ${result.signal === "BULLISH" ? "bullish" : "bearish"}`;
    dirEl.style.color = result.signal === "BULLISH" ? "var(--bullish)" : "var(--bearish)";

  } catch (err) {
    alert("Simulation error: " + err.message);
  }
}

/** Load and render Model Evaluation Benchmark cards */
async function loadBenchmarkMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    if (!res.ok) throw new Error("Could not load metrics");
    const data = await res.json();

    const container = document.getElementById("benchmark-cards");
    container.innerHTML = "";

    const results = data.model_results || {};
    const bestModel = data.best_model;

    for (const [modelName, info] of Object.entries(results)) {
      const isBest = modelName === bestModel;
      const m = info.Test_Metrics || {};

      const card = document.createElement("div");
      card.className = `model-benchmark-card ${isBest ? "highlight" : ""}`;
      card.innerHTML = `
        <div class="mb-header">
          <h4>${modelName}</h4>
          ${isBest ? '<span class="badge" style="background-color: var(--gold-primary); color: #000; font-weight:700;">★ Best Model</span>' : ''}
        </div>
        <div class="mb-stats-row">
          <div class="stat-item">
            <div class="stat-lbl">TimeSeries CV RMSE</div>
            <div class="stat-val font-mono">$${info.CV_RMSE_Mean?.toFixed(2) || "--"}</div>
          </div>
          <div class="stat-item">
            <div class="stat-lbl">Test MAE</div>
            <div class="stat-val font-mono">$${m.MAE?.toFixed(2) || "--"}</div>
          </div>
          <div class="stat-item">
            <div class="stat-lbl">Test RMSE</div>
            <div class="stat-val font-mono">$${m.RMSE?.toFixed(2) || "--"}</div>
          </div>
          <div class="stat-item">
            <div class="stat-lbl">R² Score</div>
            <div class="stat-val font-mono">${m.R2 !== undefined ? (m.R2 * 100).toFixed(1) + "%" : "--"}</div>
          </div>
        </div>
        <div class="stat-item" style="background: rgba(212, 175, 55, 0.05);">
          <div class="stat-lbl">Directional Accuracy</div>
          <div class="stat-val font-mono" style="color: var(--bullish);">${m.Directional_Accuracy_percent || "--"}%</div>
        </div>
      `;
      container.appendChild(card);
    }
  } catch (err) {
    console.warn("Metrics load error:", err);
  }
}

/** Ingest Live Data from Twelve Data API */
async function handleTwelveDataFetch() {
  const apiKey = document.getElementById("modal-api-key").value.trim();
  const interval = document.getElementById("modal-interval").value;
  const outputsize = parseInt(document.getElementById("modal-outputsize").value, 10);
  const statusEl = document.getElementById("modal-status-msg");

  if (!apiKey) {
    statusEl.className = "modal-status error";
    statusEl.innerText = "Please enter your Twelve Data API Key.";
    statusEl.classList.remove("hidden");
    return;
  }

  statusEl.className = "modal-status";
  statusEl.innerText = "Connecting to Twelve Data API & fetching market candles...";
  statusEl.classList.remove("hidden");

  try {
    const res = await fetch(`${API_BASE}/api/fetch-live`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: apiKey, interval, outputsize })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "API fetch error");

    statusEl.className = "modal-status success";
    statusEl.innerText = data.message;
    setTimeout(() => {
      document.getElementById("api-modal").classList.add("hidden");
      loadDashboardData();
      fetchLatestPrediction();
    }, 1500);

  } catch (err) {
    statusEl.className = "modal-status error";
    statusEl.innerText = err.message;
  }
}
