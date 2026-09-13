/**
 * Frontend client logic for XAU/USD Gold Price Prediction ML Dashboard
 */

// TradingView Configuration & State
const DEFAULT_TV_URL = "https://www.tradingview.com/chart/S73cSsF5/?symbol=OANDA%3AXAUUSD";
const DEFAULT_TV_SYMBOL = "OANDA:XAUUSD";

let currentTvUrl = localStorage.getItem("tv_chart_url") || DEFAULT_TV_URL;
let currentTvSymbol = localStorage.getItem("tv_chart_symbol") || DEFAULT_TV_SYMBOL;
let tvWidget = null;
let currentChartView = "ml"; // 'ml' (Candlestick Terminal) or 'tv' (TradingView Web)
let currentChartStyle = "candlestick"; // 'candlestick' or 'line'

// Lightweight Charts State (Candlestick Engine)
let lwChart = null;
let lwCandleSeries = null;
let lwLineSeries = null;
let lwSmaSeries = null;
let lwEmaSeries = null;
let lwBbUpperSeries = null;
let lwBbLowerSeries = null;

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
  // Candlestick vs Line style toggle
  const btnCandle = document.getElementById("btn-chart-candlestick");
  const btnLine = document.getElementById("btn-chart-line");
  if (btnCandle) btnCandle.addEventListener("click", () => setChartStyle("candlestick"));
  if (btnLine) btnLine.addEventListener("click", () => setChartStyle("line"));

  // Chart tab switching (Candlestick Terminal vs TradingView Web)
  document.querySelectorAll(".chart-tab-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const targetView = e.currentTarget.getAttribute("data-chart");
      switchChartView(targetView);
    });
  });

  // Responsive chart resize handler
  window.addEventListener("resize", () => {
    if (lwChart) {
      const container = document.getElementById("mlCandleChart");
      if (container && container.clientWidth > 0) {
        lwChart.applyOptions({
          width: container.clientWidth,
          height: container.clientHeight || 440
        });
      }
    }
  });

  // TradingView modal triggers
  const tvModal = document.getElementById("tv-modal");
  const btnOpenTvModal = document.getElementById("btn-open-tv-modal");
  const btnCloseTvModal = document.getElementById("btn-close-tv-modal");
  const btnCancelTvModal = document.getElementById("btn-cancel-tv-modal");
  const btnSaveTvUrl = document.getElementById("btn-save-tv-url");
  const modalTvUrlInput = document.getElementById("modal-tv-url");
  const modalTvSymbolInput = document.getElementById("modal-tv-symbol");

  if (btnOpenTvModal) {
    btnOpenTvModal.addEventListener("click", () => {
      modalTvUrlInput.value = currentTvUrl;
      modalTvSymbolInput.value = currentTvSymbol;
      const statusEl = document.getElementById("tv-modal-status-msg");
      if (statusEl) statusEl.classList.add("hidden");
      tvModal.classList.remove("hidden");
    });
  }

  if (btnCloseTvModal) btnCloseTvModal.addEventListener("click", () => tvModal.classList.add("hidden"));
  if (btnCancelTvModal) btnCancelTvModal.addEventListener("click", () => tvModal.classList.add("hidden"));

  if (modalTvUrlInput) {
    modalTvUrlInput.addEventListener("input", (e) => {
      const extracted = extractSymbolFromUrl(e.target.value.trim());
      if (extracted && modalTvSymbolInput) modalTvSymbolInput.value = extracted;
    });
  }

  if (btnSaveTvUrl) {
    btnSaveTvUrl.addEventListener("click", () => {
      const newUrl = modalTvUrlInput.value.trim() || DEFAULT_TV_URL;
      const newSymbol = modalTvSymbolInput.value.trim() || extractSymbolFromUrl(newUrl);

      currentTvUrl = newUrl;
      currentTvSymbol = newSymbol;
      localStorage.setItem("tv_chart_url", currentTvUrl);
      localStorage.setItem("tv_chart_symbol", currentTvSymbol);

      initTradingViewWidget(currentTvSymbol);

      const statusMsg = document.getElementById("tv-modal-status-msg");
      if (statusMsg) {
        statusMsg.textContent = `Connected to ${currentTvSymbol}!`;
        statusMsg.className = "modal-status success";
        statusMsg.classList.remove("hidden");
      }

      setTimeout(() => {
        tvModal.classList.add("hidden");
      }, 1000);
    });
  }

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

/** Extract symbol query parameter or fallback */
function extractSymbolFromUrl(url) {
  try {
    const parsed = new URL(url);
    const sym = parsed.searchParams.get("symbol");
    if (sym) return decodeURIComponent(sym);
  } catch (e) {
    const match = url.match(/[?&]symbol=([^&]+)/);
    if (match) return decodeURIComponent(match[1]);
  }
  return DEFAULT_TV_SYMBOL;
}

/** Initialize TradingView Advanced Chart Widget */
function initTradingViewWidget(symbol = currentTvSymbol) {
  const container = document.getElementById("tradingview_widget_container");
  if (!container) return;
  container.innerHTML = "";

  // Update DOM links to the user's specific TradingView chart URL
  const tvOpenLink = document.getElementById("tv-open-link");
  if (tvOpenLink) tvOpenLink.href = currentTvUrl;

  const btnNavTv = document.getElementById("btn-nav-tv");
  if (btnNavTv) btnNavTv.href = currentTvUrl;

  const tvBadge = document.getElementById("tv-symbol-badge");
  if (tvBadge) tvBadge.innerText = `${symbol} · 1H`;

  if (typeof TradingView !== "undefined" && TradingView.widget) {
    try {
      tvWidget = new TradingView.widget({
        autosize: true,
        symbol: symbol,
        interval: "60",
        timezone: "Etc/UTC",
        theme: "dark",
        style: "1",
        locale: "en",
        toolbar_bg: "#10141e",
        enable_publishing: false,
        allow_symbol_change: true,
        container_id: "tradingview_widget_container",
        hide_side_toolbar: false,
        withdateranges: true,
        save_image: true,
        studies: [
          "STD;SMA",
          "STD;EMA",
          "STD;Bollinger_Bands"
        ]
      });
      return;
    } catch (err) {
      console.warn("TradingView.widget initialization error, falling back to iframe embed:", err);
    }
  }

  // Fallback direct TradingView embed iframe
  container.innerHTML = `
    <iframe 
      src="https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(symbol)}&interval=60&theme=dark&style=1&timezone=Etc%2FUTC&hidesidetoolbar=0&symboledit=1&saveimage=1&toolbarbg=10141e" 
      style="width: 100%; height: 100%; border: none;" 
      allowtransparency="true" 
      scrolling="no" 
      allowfullscreen>
    </iframe>
  `;
}

/** Switch between Candlestick Terminal and TradingView live web chart */
function switchChartView(view) {
  currentChartView = view;
  const tabTv = document.getElementById("tab-btn-tv");
  const tabMl = document.getElementById("tab-btn-ml");
  const tvWrapper = document.getElementById("tv-chart-wrapper");
  const mlWrapper = document.getElementById("ml-chart-wrapper");
  const tvControls = document.getElementById("tv-controls");
  const mlControls = document.getElementById("ml-controls");
  const legendBar = document.getElementById("candle-legend-bar");

  if (view === "tv") {
    if (tabTv) tabTv.classList.add("active");
    if (tabMl) tabMl.classList.remove("active");
    if (tvWrapper) tvWrapper.classList.remove("hidden");
    if (mlWrapper) mlWrapper.classList.add("hidden");
    if (tvControls) tvControls.classList.remove("hidden");
    if (mlControls) mlControls.classList.add("hidden");
    if (legendBar) legendBar.classList.add("hidden");

    if (!tvWidget) {
      initTradingViewWidget(currentTvSymbol);
    }
  } else {
    if (tabTv) tabTv.classList.remove("active");
    if (tabMl) tabMl.classList.add("active");
    if (tvWrapper) tvWrapper.classList.add("hidden");
    if (mlWrapper) mlWrapper.classList.remove("hidden");
    if (tvControls) tvControls.classList.add("hidden");
    if (mlControls) mlControls.classList.remove("hidden");
    if (legendBar) legendBar.classList.remove("hidden");

    if (lwChart) {
      const container = document.getElementById("mlCandleChart");
      if (container && container.clientWidth > 0) {
        lwChart.applyOptions({
          width: container.clientWidth,
          height: container.clientHeight || 440
        });
        lwChart.timeScale().fitContent();
      }
    } else if (priceChart) {
      priceChart.resize();
    }
  }
}

/** Toggle between Candlestick and Line chart presentation */
function setChartStyle(style) {
  currentChartStyle = style;
  const btnCandle = document.getElementById("btn-chart-candlestick");
  const btnLine = document.getElementById("btn-chart-line");

  if (style === "candlestick") {
    if (btnCandle) btnCandle.classList.add("active");
    if (btnLine) btnLine.classList.remove("active");
    if (lwCandleSeries) lwCandleSeries.applyOptions({ visible: true });
    if (lwLineSeries) lwLineSeries.applyOptions({ visible: false });
  } else {
    if (btnCandle) btnCandle.classList.remove("active");
    if (btnLine) btnLine.classList.add("active");
    if (lwCandleSeries) lwCandleSeries.applyOptions({ visible: false });
    if (lwLineSeries) lwLineSeries.applyOptions({ visible: true });
  }
}

/** Update the floating OHLC inspection bar */
function updateLegendBar(bar) {
  if (!bar) return;
  const o = typeof bar.open === "number" ? bar.open : (bar.value ?? 0);
  const h = typeof bar.high === "number" ? bar.high : o;
  const l = typeof bar.low === "number" ? bar.low : o;
  const c = typeof bar.close === "number" ? bar.close : (bar.value ?? o);
  const diff = c - o;
  const pct = o !== 0 ? (diff / o) * 100 : 0;
  const isUp = diff >= 0;

  const openEl = document.getElementById("legend-open");
  const highEl = document.getElementById("legend-high");
  const lowEl = document.getElementById("legend-low");
  const closeEl = document.getElementById("legend-close");
  const chgEl = document.getElementById("legend-change");

  if (openEl) openEl.innerText = `$${o.toFixed(2)}`;
  if (highEl) highEl.innerText = `$${h.toFixed(2)}`;
  if (lowEl) lowEl.innerText = `$${l.toFixed(2)}`;
  if (closeEl) closeEl.innerText = `$${c.toFixed(2)}`;
  if (chgEl) {
    chgEl.innerText = `${isUp ? "+" : ""}$${diff.toFixed(2)} (${isUp ? "+" : ""}${pct.toFixed(2)}%)`;
    chgEl.className = isUp ? "bullish-text" : "bearish-text";
  }
}

/** Safely convert date string into Unix seconds timestamp */
function parseCandleTime(dtStr) {
  if (!dtStr) return 0;
  const clean = dtStr.trim();
  const iso = clean.includes("T") ? clean : clean.replace(" ", "T") + "Z";
  const ms = new Date(iso).getTime();
  if (isNaN(ms)) {
    const fallback = new Date(clean).getTime();
    return isNaN(fallback) ? 0 : Math.floor(fallback / 1000);
  }
  return Math.floor(ms / 1000);
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

/** Render Main Interactive Candlestick Chart with Technical Overlays */
function renderPriceChart(candles) {
  if (!candles || candles.length === 0) return;

  // Prefer Lightweight Charts (Financial Candlestick Engine)
  if (typeof LightweightCharts !== "undefined") {
    const container = document.getElementById("mlCandleChart");
    if (!container) return;

    // Filter, deduplicate, and sort candles by time
    const seenTimes = new Set();
    const sortedCandles = [];
    for (const c of candles) {
      const t = parseCandleTime(c.datetime);
      if (t > 0 && !seenTimes.has(t)) {
        seenTimes.add(t);
        sortedCandles.push({
          ...c,
          _time: t,
          open: Number(c.open),
          high: Number(c.high),
          low: Number(c.low),
          close: Number(c.close)
        });
      }
    }
    sortedCandles.sort((a, b) => a._time - b._time);

    const candleData = sortedCandles.map(c => ({
      time: c._time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close
    }));

    const lineData = sortedCandles.map(c => ({
      time: c._time,
      value: c.close
    }));

    const smaData = sortedCandles
      .filter(c => c.sma_20 !== undefined && c.sma_20 !== null)
      .map(c => ({ time: c._time, value: Number(c.sma_20) }));

    const emaData = sortedCandles
      .filter(c => c.ema_50 !== undefined && c.ema_50 !== null)
      .map(c => ({ time: c._time, value: Number(c.ema_50) }));

    const bbUpperData = sortedCandles
      .filter(c => c.bb_upper !== undefined && c.bb_upper !== null)
      .map(c => ({ time: c._time, value: Number(c.bb_upper) }));

    const bbLowerData = sortedCandles
      .filter(c => c.bb_lower !== undefined && c.bb_lower !== null)
      .map(c => ({ time: c._time, value: Number(c.bb_lower) }));

    if (!lwChart) {
      container.innerHTML = "";
      lwChart = LightweightCharts.createChart(container, {
        width: container.clientWidth || 800,
        height: 440,
        layout: {
          background: { type: "solid", color: "#10141e" },
          textColor: "#94a3b8",
          fontSize: 12,
          fontFamily: "'JetBrains Mono', monospace"
        },
        grid: {
          vertLines: { color: "rgba(255, 255, 255, 0.04)" },
          horzLines: { color: "rgba(255, 255, 255, 0.04)" }
        },
        crosshair: {
          mode: LightweightCharts.CrosshairMode.Normal,
          vertLine: {
            color: "rgba(212, 175, 55, 0.4)",
            width: 1,
            style: LightweightCharts.LineStyle.Dashed
          },
          horzLine: {
            color: "rgba(212, 175, 55, 0.4)",
            width: 1,
            style: LightweightCharts.LineStyle.Dashed
          }
        },
        rightPriceScale: {
          borderColor: "rgba(255, 255, 255, 0.1)",
          scaleMargins: { top: 0.08, bottom: 0.08 }
        },
        timeScale: {
          borderColor: "rgba(255, 255, 255, 0.1)",
          timeVisible: true,
          secondsVisible: false
        },
        handleScroll: true,
        handleScale: true
      });

      // 1. Candlestick series (green/red candles & wicks)
      lwCandleSeries = lwChart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: "#00c087",
        downColor: "#ef4444",
        borderVisible: false,
        wickUpColor: "#00c087",
        wickDownColor: "#ef4444"
      });

      // 2. Line series (optional fallback/alternate style)
      lwLineSeries = lwChart.addSeries(LightweightCharts.LineSeries, {
        color: "#d4af37",
        lineWidth: 2,
        visible: false
      });

      // 3. Technical Indicator Overlays
      lwSmaSeries = lwChart.addSeries(LightweightCharts.LineSeries, {
        color: "#00b4d8",
        lineWidth: 1.5,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        title: "SMA 20"
      });

      lwEmaSeries = lwChart.addSeries(LightweightCharts.LineSeries, {
        color: "#9d4edd",
        lineWidth: 1.5,
        title: "EMA 50"
      });

      lwBbUpperSeries = lwChart.addSeries(LightweightCharts.LineSeries, {
        color: "rgba(255, 255, 255, 0.35)",
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        title: "BB Upper"
      });

      lwBbLowerSeries = lwChart.addSeries(LightweightCharts.LineSeries, {
        color: "rgba(255, 255, 255, 0.35)",
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        title: "BB Lower"
      });

      // Crosshair inspection updates legend bar
      lwChart.subscribeCrosshairMove(param => {
        if (!param || !param.time || !param.seriesData) {
          if (sortedCandles.length > 0) {
            updateLegendBar(sortedCandles[sortedCandles.length - 1]);
          }
          return;
        }
        const candleBar = param.seriesData.get(lwCandleSeries);
        const lineBar = param.seriesData.get(lwLineSeries);
        if (candleBar) {
          updateLegendBar(candleBar);
        } else if (lineBar) {
          updateLegendBar(lineBar);
        }
      });
    }

    // Set series data
    lwCandleSeries.setData(candleData);
    lwLineSeries.setData(lineData);
    lwSmaSeries.setData(smaData);
    lwEmaSeries.setData(emaData);
    lwBbUpperSeries.setData(bbUpperData);
    lwBbLowerSeries.setData(bbLowerData);

    // Apply overlay visibility & style
    updateChartDatasets();
    setChartStyle(currentChartStyle);

    // Auto-fit visible bars
    lwChart.timeScale().fitContent();

    // Set legend bar to latest candle
    if (sortedCandles.length > 0) {
      updateLegendBar(sortedCandles[sortedCandles.length - 1]);
    }

    return;
  }

  // Fallback: Chart.js Canvas
  renderChartJsFallback(candles);
}

/** Fallback renderer using Chart.js */
function renderChartJsFallback(candles) {
  const canvas = document.getElementById("priceChart");
  if (!canvas) return;
  canvas.classList.remove("hidden");
  const ctx = canvas.getContext("2d");
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
      pointRadius: 0
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
        interaction: { mode: "index", intersect: false }
      }
    });
  }
}

/** Update overlay visibility according to user toggles */
function updateChartDatasets() {
  const showSma = document.getElementById("toggle-sma") ? document.getElementById("toggle-sma").checked : true;
  const showEma = document.getElementById("toggle-ema") ? document.getElementById("toggle-ema").checked : true;
  const showBb = document.getElementById("toggle-bb") ? document.getElementById("toggle-bb").checked : true;

  if (lwChart) {
    if (lwSmaSeries) lwSmaSeries.applyOptions({ visible: showSma });
    if (lwEmaSeries) lwEmaSeries.applyOptions({ visible: showEma });
    if (lwBbUpperSeries) lwBbUpperSeries.applyOptions({ visible: showBb });
    if (lwBbLowerSeries) lwBbLowerSeries.applyOptions({ visible: showBb });
  }

  if (priceChart && priceChart.data && priceChart.data.datasets) {
    if (priceChart.data.datasets[1]) priceChart.data.datasets[1].hidden = !showSma;
    if (priceChart.data.datasets[2]) priceChart.data.datasets[2].hidden = !showEma;
    if (priceChart.data.datasets[3]) priceChart.data.datasets[3].hidden = !showBb;
    if (priceChart.data.datasets[4]) priceChart.data.datasets[4].hidden = !showBb;
    priceChart.update();
  }
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
