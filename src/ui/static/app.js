"use strict";

/* ============================================================
 * Live Trading Monitor — frontend logic
 * Reads WS contract:
 *   ws://<host>/ws/overview            -> {type:"overview", ...}
 *   ws://<host>/ws/strategy/{id}       -> {type:"detail", ...} or {type:"error", message}
 * No build step, no dependencies besides Plotly (CDN, loaded in index.html).
 * ============================================================ */

// ---------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------

const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
const wsBase = `${wsProtocol}//${location.host}`;

function fmtNum(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "--";
  return Number(v).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtInt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "--";
  return Number(v).toLocaleString();
}

function fmtPct(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "--";
  return `${(Number(v) * 100).toFixed(digits)}%`;
}

function pnlClass(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  if (v > 0) return "pnl-pos";
  if (v < 0) return "pnl-neg";
  return "pnl-zero";
}

function signedFmt(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "--";
  const s = Number(v) > 0 ? "+" : "";
  return `${s}${fmtNum(v, digits)}`;
}

function fmtUptime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "--";
  seconds = Math.floor(seconds);
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function fmtClock(iso) {
  if (!iso) return "--:--:--";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "--:--:--";
  return d.toUTCString().split(" ")[4] + " UTC";
}

function fmtTime(iso) {
  if (!iso) return "--";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toISOString().replace("T", " ").replace("Z", "").slice(0, 19);
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function dirLabel(direction) {
  if (direction === 1) return { text: "LONG", cls: "dir-long" };
  if (direction === -1) return { text: "SHORT", cls: "dir-short" };
  return { text: "--", cls: "dir-flat" };
}

// Classify a trades.csv row into a badge + row style, so a scan down the
// table instantly separates "just a signal, nothing happened" from "a
// position was actually opened/closed" — and, for exits, whether it closed
// in profit or loss. Reuses the app's existing green/red/grey/blue tokens
// (see .evt-* in style.css) rather than a new palette.
// Gross (fee-free) mark-to-market estimate — labeled "unrealized" everywhere
// it's shown, distinct from the fee-netted `realized_pnl` in trades.csv, so
// no one mistakes it for the exact number the exchange would report on close.
function unrealizedPnl(direction, entryPrice, sizeRemaining, markPrice) {
  if (direction === null || direction === undefined) return null;
  if (entryPrice === null || entryPrice === undefined) return null;
  if (sizeRemaining === null || sizeRemaining === undefined) return null;
  if (markPrice === null || markPrice === undefined) return null;
  return direction * sizeRemaining * (markPrice - entryPrice);
}

function currentMarkPrice() {
  return market.lastCandle ? market.lastCandle.close : null;
}

// Shared $/% conversion for every equity/PnL/returns chart, driven by the
// single global toggle in the topbar (state.pnlMode). Percent mode always
// expresses "return since the first point in this series" — each series
// (per-strategy equity curve, per-trade PnL, etc.) uses its own baseline,
// so strategies/trades at different capital sizes stay comparable.
function equityToDisplay(points) {
  if (state.pnlMode !== "percent" || !points.length) return points;
  const base = points[0].e;
  if (!base) return points.map((p) => ({ t: p.t, e: 0 }));
  return points.map((p) => ({ t: p.t, e: ((p.e / base) - 1) * 100 }));
}

function amountToDisplay(value, baseline) {
  if (state.pnlMode !== "percent" || !baseline) return value;
  return (value / baseline) * 100;
}

function fmtDisplayValue(value) {
  return state.pnlMode === "percent" ? `${value >= 0 ? "+" : ""}${value.toFixed(2)}%` : signedFmt(value);
}

function eventInfo(event, direction, pnlNet) {
  if (event === "SIGNAL") {
    return { text: "SIGNAL", cls: "evt-signal", row: "row-signal" };
  }
  if (event === "ENTRY") {
    const dir = dirLabel(direction);
    const cls = dir.cls === "dir-long" ? "evt-entry-long" : "evt-entry-short";
    return { text: `ENTRY ${dir.text}`, cls, row: "row-trade" };
  }
  // EXIT_* / PARTIAL_* — color by realized P&L, not by exit reason, since
  // e.g. a time-stop or even a stop-loss (after a move-to-breakeven) can
  // close flat or in profit just as often as at a loss.
  let cls = "evt-exit-flat";
  if (pnlNet !== null && pnlNet !== undefined && !Number.isNaN(pnlNet)) {
    if (pnlNet > 0) cls = "evt-exit-win";
    else if (pnlNet < 0) cls = "evt-exit-loss";
  }
  return { text: String(event || "").replace(/_/g, " "), cls, row: "row-trade" };
}

// Consistent color per strategy_id for the aggregate chart.
const PALETTE = [
  "#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff",
  "#39c5cf", "#f778ba", "#ffa657", "#79c0ff", "#7ee787",
];
const colorCache = new Map();
function colorFor(id) {
  if (!colorCache.has(id)) {
    colorCache.set(id, PALETTE[colorCache.size % PALETTE.length]);
  }
  return colorCache.get(id);
}

// ---------------------------------------------------------------
// Reconnecting WebSocket wrapper
// ---------------------------------------------------------------

class ReconnectingSocket {
  /**
   * @param {string} url
   * @param {(data: any) => void} onMessage
   * @param {(state: "connecting"|"open"|"reconnecting"|"closed") => void} onStateChange
   */
  constructor(url, onMessage, onStateChange) {
    this.url = url;
    this.onMessage = onMessage;
    this.onStateChange = onStateChange || (() => {});
    this.ws = null;
    this.attempt = 0;
    this.closedByUser = false;
    this.reconnectTimer = null;
    this._connect();
  }

  _connect() {
    if (this.closedByUser) return;
    this.onStateChange(this.attempt === 0 ? "connecting" : "reconnecting");
    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch (e) {
      this._scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.attempt = 0;
      this.onStateChange("open");
    };

    ws.onmessage = (evt) => {
      let data;
      try {
        data = JSON.parse(evt.data);
      } catch (e) {
        return;
      }
      this.onMessage(data);
    };

    ws.onclose = () => {
      if (this.closedByUser) {
        this.onStateChange("closed");
        return;
      }
      this._scheduleReconnect();
    };

    ws.onerror = () => {
      // onclose will fire right after; let it handle reconnection.
      try { ws.close(); } catch (e) { /* noop */ }
    };
  }

  _scheduleReconnect() {
    this.attempt += 1;
    const delay = Math.min(1000 * Math.pow(2, this.attempt - 1), 10000);
    this.onStateChange("reconnecting");
    clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => this._connect(), delay);
  }

  close() {
    this.closedByUser = true;
    clearTimeout(this.reconnectTimer);
    if (this.ws) {
      try { this.ws.close(); } catch (e) { /* noop */ }
    }
  }
}

// ---------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------

const el = {
  hamburger: document.getElementById("hamburger"),
  sidebar: document.getElementById("sidebar"),
  sidebarBackdrop: document.getElementById("sidebar-backdrop"),
  navOverview: document.getElementById("nav-overview"),
  strategyList: document.getElementById("strategy-list"),
  connStatus: document.getElementById("conn-status"),
  connLabel: document.querySelector("#conn-status .conn-label"),
  serverClock: document.getElementById("server-clock"),

  viewOverview: document.getElementById("view-overview"),
  viewDetail: document.getElementById("view-detail"),

  statTotalPnl: document.getElementById("stat-total-pnl"),
  statOpenPositions: document.getElementById("stat-open-positions"),
  statTradesToday: document.getElementById("stat-trades-today"),
  overviewTableBody: document.getElementById("overview-table-body"),
  overviewChart: document.getElementById("overview-chart"),

  marketChart: document.getElementById("market-chart"),
  marketPanel: document.getElementById("market-panel"),
  marketConn: document.getElementById("market-conn"),
  marketConnLabel: document.querySelector("#market-conn .conn-label"),
  marketLastPrice: document.getElementById("market-last-price"),
  marketLastChange: document.getElementById("market-last-change"),
  tfSelector: document.getElementById("tf-selector"),

  detailError: document.getElementById("detail-error"),
  detailHeader: document.getElementById("detail-header"),
  detailTitle: document.getElementById("detail-title"),
  detailBadges: document.getElementById("detail-badges"),
  detailUptime: document.getElementById("detail-uptime"),
  detailCpu: document.getElementById("detail-cpu"),
  detailRss: document.getElementById("detail-rss"),
  positionBody: document.getElementById("position-body"),
  statsBody: document.getElementById("stats-body"),
  analysisBody: document.getElementById("analysis-body"),
  analysisAge: document.getElementById("analysis-age"),
  detailChart: document.getElementById("detail-chart"),
  drawdownChart: document.getElementById("drawdown-chart"),
  calendarHeatmap: document.getElementById("calendar-heatmap"),
  rMultipleChart: document.getElementById("r-multiple-chart"),
  sessionChart: document.getElementById("session-chart"),
  tradesTableBody: document.getElementById("trades-table-body"),
  signalsListBody: document.getElementById("signals-list-body"),
  logTail: document.getElementById("log-tail"),

  tradeModal: document.getElementById("trade-modal"),
  tradeModalTitle: document.getElementById("trade-modal-title"),
  tradeModalSubtitle: document.getElementById("trade-modal-subtitle"),
  tradeModalClose: document.getElementById("trade-modal-close"),
  tradeModalChart: document.getElementById("trade-modal-chart"),
  tradeModalPnl: document.getElementById("trade-modal-pnl"),
  tradeModalDrawdown: document.getElementById("trade-modal-drawdown"),

  pnlModeToggle: document.getElementById("pnl-mode-toggle"),
};

// ---------------------------------------------------------------
// App state
// ---------------------------------------------------------------

const state = {
  currentView: "overview", // "overview" | strategy_id
  strategiesById: new Map(), // latest overview entries, in original order
  strategyOrder: [],
  overviewChartInitialized: false,
  detailChartInitialized: false,
  drawdownChartInitialized: false,
  rMultipleChartInitialized: false,
  sessionChartInitialized: false,
  detailSocket: null,
  lastLogLines: [],
  lastDetailStrategyId: null,
  lastAnalysisTimestamp: null,
  currentPosition: null, // detail view's open position, cached for tick-driven unrealized P&L
  tradeEpisodes: [], // detail view's ENTRY->close groupings, cached for the trade-replay modal

  pnlMode: "currency", // "currency" | "percent" — shared by every PnL/equity/returns chart
  lastOverviewMsg: null, // cached so toggling pnlMode can re-render without waiting for the next tick
  lastDetailMsg: null,
  detailStartingEquity: null, // % baseline for the current detail view's charts
};

// ---------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------

function statusBadgeInfo(strat) {
  if (strat.stalled) return { cls: "stalled", label: "STALLED" };
  if (strat.status === "running") return { cls: "running", label: "RUNNING" };
  if (strat.status === "stopped") return { cls: "stopped", label: "STOPPED" };
  return { cls: "unknown", label: "UNKNOWN" };
}

function renderSidebar() {
  const frag = document.createDocumentFragment();
  for (const id of state.strategyOrder) {
    const strat = state.strategiesById.get(id);
    if (!strat) continue;
    const info = statusBadgeInfo(strat);
    const item = document.createElement("div");
    item.className = "strategy-item" + (state.currentView === id ? " active" : "");
    item.dataset.id = id;
    item.innerHTML = `
      <div class="strategy-item-row">
        <span class="status-dot ${info.cls}" title="${info.label}"></span>
        <span class="strategy-item-name">${escapeHtml(id)}</span>
      </div>
      <div class="strategy-item-meta">${escapeHtml(strat.scenario || "")} &middot; ${escapeHtml(strat.exchange || "")}</div>
    `;
    item.addEventListener("click", () => selectStrategy(id));
    frag.appendChild(item);
  }
  el.strategyList.innerHTML = "";
  el.strategyList.appendChild(frag);
}

function closeSidebarMobile() {
  el.sidebar.classList.remove("open");
  el.sidebarBackdrop.classList.remove("open");
}

el.hamburger.addEventListener("click", () => {
  el.sidebar.classList.toggle("open");
  el.sidebarBackdrop.classList.toggle("open");
});
el.sidebarBackdrop.addEventListener("click", closeSidebarMobile);

el.navOverview.addEventListener("click", () => selectOverview());

function setPnlMode(mode) {
  if (state.pnlMode === mode) return;
  state.pnlMode = mode;
  el.pnlModeToggle.querySelectorAll(".pnl-toggle-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === mode);
  });

  if (state.currentView === "overview" && state.lastOverviewMsg) {
    renderOverviewPanel(state.lastOverviewMsg);
  } else if (state.lastDetailMsg) {
    renderDetailPanel(state.lastDetailMsg);
  }
}

el.pnlModeToggle.addEventListener("click", (e) => {
  const btn = e.target.closest(".pnl-toggle-btn");
  if (btn) setPnlMode(btn.dataset.mode);
});

// ---------------------------------------------------------------
// View switching
// ---------------------------------------------------------------

function selectOverview() {
  state.currentView = "overview";
  el.navOverview.classList.add("active");
  el.viewOverview.classList.remove("hidden");
  el.viewDetail.classList.add("hidden");
  closeSidebarMobile();
  teardownDetailSocket();
  renderSidebar();
}

function selectStrategy(id) {
  state.currentView = id;
  el.navOverview.classList.remove("active");
  el.viewOverview.classList.add("hidden");
  el.viewDetail.classList.remove("hidden");
  closeSidebarMobile();
  renderSidebar();

  if (state.lastDetailStrategyId !== id) {
    teardownDetailSocket();
    state.detailChartInitialized = false;
    state.drawdownChartInitialized = false;
    state.rMultipleChartInitialized = false;
    state.sessionChartInitialized = false;
    state.lastLogLines = [];
    state.lastDetailStrategyId = id;
    el.detailError.classList.add("hidden");
    el.detailTitle.textContent = id;
    setupDetailSocket(id);
  }
}

function teardownDetailSocket() {
  if (state.detailSocket) {
    state.detailSocket.close();
    state.detailSocket = null;
  }
}

function setupDetailSocket(id) {
  const url = `${wsBase}/ws/strategy/${encodeURIComponent(id)}`;
  state.detailSocket = new ReconnectingSocket(
    url,
    (data) => handleDetailMessage(id, data),
    () => { /* per-detail connection state currently not surfaced separately */ }
  );
}

// ---------------------------------------------------------------
// Overview feed
// ---------------------------------------------------------------

function setConnState(connState) {
  el.connStatus.classList.remove("reconnecting", "down");
  if (connState === "open") {
    el.connLabel.textContent = "live";
  } else if (connState === "reconnecting") {
    el.connStatus.classList.add("reconnecting");
    el.connLabel.textContent = "reconnecting…";
  } else if (connState === "connecting") {
    el.connStatus.classList.add("reconnecting");
    el.connLabel.textContent = "connecting…";
  } else {
    el.connStatus.classList.add("down");
    el.connLabel.textContent = "disconnected";
  }
}

function handleOverviewMessage(msg) {
  if (msg.type !== "overview") return;

  state.lastOverviewMsg = msg;
  el.serverClock.textContent = fmtClock(msg.server_time);

  state.strategyOrder = (msg.strategies || []).map((s) => s.strategy_id);
  state.strategiesById = new Map((msg.strategies || []).map((s) => [s.strategy_id, s]));
  renderSidebar();

  if (state.currentView === "overview") {
    renderOverviewPanel(msg);
  }
}

function renderOverviewPanel(msg) {
  const agg = msg.aggregate || {};

  el.statTotalPnl.textContent = signedFmt(agg.total_realized_pnl);
  el.statTotalPnl.className = "stat-value " + pnlClass(agg.total_realized_pnl);
  el.statOpenPositions.textContent = fmtInt(agg.open_positions);
  el.statTradesToday.textContent = fmtInt(agg.trades_today);

  // Summary table
  const rows = (msg.strategies || []).map((s) => {
    const info = statusBadgeInfo(s);
    const dryBadge = s.dry_run
      ? `<span class="badge badge-dryrun">DRY</span>`
      : `<span class="badge badge-live">LIVE</span>`;
    const posCls = /short/i.test(s.position_desc || "") ? "dir-short"
      : /long/i.test(s.position_desc || "") ? "dir-long" : "dir-flat";
    const upnl = unrealizedPnl(s.position_direction, s.position_entry_price,
      s.position_size_remaining, currentMarkPrice());
    const upnlHtml = upnl === null
      ? `<span class="text-faint">--</span>`
      : `<span class="${pnlClass(upnl)}">${signedFmt(upnl)}</span>`;
    return `
      <tr class="clickable" data-id="${escapeHtml(s.strategy_id)}">
        <td class="mono">${escapeHtml(s.strategy_id)}</td>
        <td>${escapeHtml(s.scenario || "")}</td>
        <td>${escapeHtml(s.exchange || "")}</td>
        <td>${dryBadge}</td>
        <td><span class="badge badge-${info.cls}">${info.label}</span></td>
        <td class="${posCls}">${escapeHtml(s.position_desc || "flat")}</td>
        <td class="${pnlClass(s.realized_pnl)}">${signedFmt(s.realized_pnl)}</td>
        <td class="upnl-cell" data-upnl-row="${escapeHtml(s.strategy_id)}">${upnlHtml}</td>
        <td>${fmtNum(s.equity)}</td>
      </tr>
    `;
  });
  el.overviewTableBody.innerHTML = rows.length
    ? rows.join("")
    : `<tr><td colspan="9" class="empty-row">No strategies reported.</td></tr>`;

  el.overviewTableBody.querySelectorAll("tr.clickable").forEach((tr) => {
    tr.addEventListener("click", () => selectStrategy(tr.dataset.id));
  });

  renderOverviewChart(agg.equity_curves || {});
}

function renderOverviewChart(equityCurves) {
  const ids = Object.keys(equityCurves);
  const isPct = state.pnlMode === "percent";
  const traces = ids.map((id) => {
    const pts = equityToDisplay(equityCurves[id] || []);
    return {
      x: pts.map((p) => p.t),
      y: pts.map((p) => p.e),
      type: "scattergl",
      mode: "lines",
      name: id,
      line: { color: colorFor(id), width: 1.6 },
      hovertemplate: (isPct ? "%{y:.2f}%" : "%{y:.2f}") + "<br>%{x}<extra>" + id + "</extra>",
    };
  });

  const layout = baseChartLayout();
  layout.yaxis = { ...layout.yaxis, title: isPct ? "Return %" : "Equity" };

  if (!traces.length) {
    traces.push({ x: [], y: [], type: "scattergl", mode: "lines", name: "no data" });
  }

  if (!state.overviewChartInitialized) {
    Plotly.newPlot(el.overviewChart, traces, layout, plotlyConfig());
    state.overviewChartInitialized = true;
  } else {
    Plotly.react(el.overviewChart, traces, layout, plotlyConfig());
  }
}

// ---------------------------------------------------------------
// Detail feed
// ---------------------------------------------------------------

function handleDetailMessage(requestedId, msg) {
  // Ignore stale messages if the user already switched to another strategy.
  if (state.lastDetailStrategyId !== requestedId) return;

  if (msg.type === "error") {
    el.detailError.textContent = `Error: ${msg.message || "unknown error"}`;
    el.detailError.classList.remove("hidden");
    return;
  }
  if (msg.type !== "detail") return;

  el.detailError.classList.add("hidden");
  renderDetailPanel(msg);
}

function renderDetailPanel(msg) {
  state.lastDetailMsg = msg;
  const meta = msg.meta || {};
  const curve = msg.equity_curve || [];
  state.detailStartingEquity = curve.length ? curve[0].e : (meta.capital || null);
  el.detailTitle.textContent = `${msg.strategy_id}`;

  // Badges: status/stalled, dry-run, exchange, scenario
  const info = statusBadgeInfo(msg);
  const badges = [`<span class="badge badge-${info.cls}">${info.label}</span>`];
  badges.push(meta.dry_run
    ? `<span class="badge badge-dryrun">DRY-RUN</span>`
    : `<span class="badge badge-live">LIVE</span>`);
  if (meta.scenario) badges.push(`<span class="badge">${escapeHtml(meta.scenario)}</span>`);
  if (meta.exchange) badges.push(`<span class="badge">${escapeHtml(meta.exchange)}</span>`);
  el.detailBadges.innerHTML = badges.join("");

  el.detailUptime.textContent = fmtUptime(msg.uptime_seconds);
  el.detailCpu.textContent = msg.cpu_percent === null || msg.cpu_percent === undefined
    ? "--" : `${fmtNum(msg.cpu_percent, 1)}%`;
  el.detailRss.textContent = msg.rss_mb === null || msg.rss_mb === undefined
    ? "--" : `${fmtNum(msg.rss_mb, 1)} MB`;

  state.tradeEpisodes = msg.trade_episodes || [];

  renderPosition(msg.position || { active: false });
  renderStats(msg.stats || { insufficient_data: true });
  renderAnalysis(msg.analysis || {});
  renderDetailChart(msg.equity_curve || []);
  renderDrawdownChart(msg.equity_curve || []);
  renderCalendarHeatmap(msg.daily_pnl || {});
  renderRMultipleHistogram(msg.r_multiples || []);
  renderSessionStats(msg.session_stats || []);
  renderSignalsList(msg.trades || []);
  renderTradesTable(buildTradeRows());
  renderLogTail(msg.log_tail || []);
}

// ---------------------------------------------------------------
// Latest Analysis card
// ---------------------------------------------------------------

function fmtRelativeAge(iso) {
  if (!iso) return "--";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "--";
  const seconds = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (seconds < 60) return `updated ${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `updated ${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `updated ${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `updated ${days}d ago`;
}

function metricValueClass(key, value) {
  if (typeof value === "boolean") return "";
  const k = key.toLowerCase();
  if (typeof value === "number") {
    // Fields that are directionally signed (positive=bullish, negative=bearish).
    if (/^(composite|obv_trend)$/.test(k)) {
      if (value > 0) return "val-pos";
      if (value < 0) return "val-neg";
      return "val-neutral";
    }
  }
  if (k === "regime") {
    if (/long|bull|up/i.test(String(value))) return "val-pos";
    if (/short|bear|down/i.test(String(value))) return "val-neg";
    return "val-neutral";
  }
  return "";
}

function fmtMetricValue(value) {
  if (value === null || value === undefined) return "--";
  if (typeof value === "boolean") return value ? "✓ Yes" : "✗ No";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  return escapeHtml(String(value));
}

function renderAnalysis(analysis) {
  if (!analysis || Object.keys(analysis).length === 0) {
    el.analysisBody.innerHTML = `<div class="empty-row">Waiting for first analysis…</div>`;
    el.analysisAge.textContent = "--";
    state.lastAnalysisTimestamp = null;
    return;
  }

  state.lastAnalysisTimestamp = analysis.timestamp || null;
  el.analysisAge.textContent = fmtRelativeAge(analysis.timestamp);

  const signalCls = analysis.signal > 0 ? "signal-long" : analysis.signal < 0 ? "signal-short" : "";
  const reasonHtml = `<div class="analysis-reason ${signalCls}">${escapeHtml(analysis.reason || "--")}</div>`;

  const metaBits = [];
  if (analysis.close !== undefined) metaBits.push(`<span>Close: ${fmtNum(analysis.close)}</span>`);
  if (analysis.composite !== undefined) {
    metaBits.push(`<span class="${pnlClass(analysis.composite)}">Composite: ${signedFmt(analysis.composite)}</span>`);
  }
  if (analysis.atr !== undefined) metaBits.push(`<span>ATR: ${fmtNum(analysis.atr)}</span>`);
  const metaHtml = metaBits.length ? `<div class="analysis-meta-row">${metaBits.join("")}</div>` : "";

  const metrics = analysis.metrics || {};
  const metricEntries = Object.entries(metrics);
  const metricsHtml = metricEntries.length
    ? `<div class="metrics-grid">${metricEntries.map(([k, v]) => `
        <div class="mg-item">
          <span class="mg-label">${escapeHtml(k.replace(/_/g, " "))}</span>
          <span class="mg-value ${metricValueClass(k, v)}">${fmtMetricValue(v)}</span>
        </div>
      `).join("")}</div>`
    : "";

  el.analysisBody.innerHTML = reasonHtml + metaHtml + metricsHtml;
}

function renderPosition(pos) {
  state.currentPosition = (pos && pos.active) ? pos : null;

  if (!pos || !pos.active) {
    el.positionBody.innerHTML = `<div class="position-flat">flat — no open position</div>`;
    return;
  }
  const dir = dirLabel(pos.direction);
  const tpRow = (label, price, hit) => `
    <div class="pg-item">
      <span class="pg-label">${label}${hit ? " ✓" : ""}</span>
      <span class="${hit ? "tp-hit" : "tp-open"}">${fmtNum(price)}</span>
    </div>`;
  const upnl = unrealizedPnl(pos.direction, pos.entry_price, pos.size_remaining, currentMarkPrice());

  el.positionBody.innerHTML = `
    <div class="position-grid">
      <div class="pg-item"><span class="pg-label">Direction</span><span class="${dir.cls}">${dir.text}</span></div>
      <div class="pg-item"><span class="pg-label">Entry Price</span><span>${fmtNum(pos.entry_price)}</span></div>
      <div class="pg-item"><span class="pg-label">Size (total)</span><span>${fmtNum(pos.size_total, 4)}</span></div>
      <div class="pg-item"><span class="pg-label">Size (remaining)</span><span>${fmtNum(pos.size_remaining, 4)}</span></div>
      <div class="pg-item"><span class="pg-label">Stop Loss</span><span>${fmtNum(pos.sl)}</span></div>
      <div class="pg-item"><span class="pg-label">Break-even moved</span><span>${pos.be_moved ? "yes" : "no"}</span></div>
      ${tpRow("TP1", pos.tp1, pos.tp1_hit)}
      ${tpRow("TP2", pos.tp2, pos.tp2_hit)}
      ${tpRow("TP3", pos.tp3, false)}
      <div class="pg-item"><span class="pg-label">Entry Time</span><span>${fmtTime(pos.entry_time)}</span></div>
      <div class="pg-item"><span class="pg-label">Composite</span><span>${fmtNum(pos.composite)}</span></div>
      <div class="pg-item"><span class="pg-label">ATR @ Entry</span><span>${fmtNum(pos.atr_at_entry)}</span></div>
      <div class="pg-item"><span class="pg-label">Realized PnL</span><span class="${pnlClass(pos.realized_pnl)}">${signedFmt(pos.realized_pnl)}</span></div>
      <div class="pg-item"><span class="pg-label">Unrealized PnL (live)</span><span id="detail-upnl" class="${pnlClass(upnl)}">${upnl === null ? "--" : signedFmt(upnl)}</span></div>
    </div>
  `;
}

// Called on every live market price tick (far more frequent than the 2s
// WS payload refresh) so unrealized P&L actually feels "live" rather than
// stepping every couple of seconds. Updates existing DOM text in place —
// never rebuilds a table/card on a price tick.
function refreshUnrealizedPnl() {
  const mark = currentMarkPrice();
  if (mark === null) return;

  if (state.currentView === "overview") {
    for (const id of state.strategyOrder) {
      const s = state.strategiesById.get(id);
      if (!s) continue;
      const cell = el.overviewTableBody.querySelector(`.upnl-cell[data-upnl-row="${CSS.escape(id)}"] span`);
      if (!cell) continue;
      const upnl = unrealizedPnl(s.position_direction, s.position_entry_price, s.position_size_remaining, mark);
      if (upnl === null) continue;
      cell.textContent = signedFmt(upnl);
      cell.className = pnlClass(upnl);
    }
  } else if (state.currentPosition) {
    const el2 = document.getElementById("detail-upnl");
    if (!el2) return;
    const pos = state.currentPosition;
    const upnl = unrealizedPnl(pos.direction, pos.entry_price, pos.size_remaining, mark);
    if (upnl === null) return;
    el2.textContent = signedFmt(upnl);
    el2.className = pnlClass(upnl);
  }
}

function renderStats(stats) {
  if (!stats || stats.insufficient_data) {
    el.statsBody.innerHTML = `<div class="empty-row">Insufficient data yet.</div>`;
    return;
  }
  el.statsBody.innerHTML = `
    <div class="stats-grid">
      <div class="sg-item"><span class="sg-label">Trades</span><span>${fmtInt(stats.n_trades)}</span></div>
      <div class="sg-item"><span class="sg-label">Win Rate</span><span>${fmtPct(stats.win_rate)}</span></div>
      <div class="sg-item"><span class="sg-label">Avg Win</span><span class="pnl-pos">${signedFmt(stats.avg_win)}</span></div>
      <div class="sg-item"><span class="sg-label">Avg Loss</span><span class="pnl-neg">${signedFmt(stats.avg_loss)}</span></div>
      <div class="sg-item"><span class="sg-label">Max Drawdown</span><span class="pnl-neg">${fmtPct(stats.max_drawdown)}</span></div>
      <div class="sg-item"><span class="sg-label">Sharpe</span><span>${fmtNum(stats.sharpe)}</span></div>
      <div class="sg-item"><span class="sg-label">Profit Factor</span><span>${stats.profit_factor === null || stats.profit_factor === undefined ? "--" : fmtNum(stats.profit_factor, 2)}</span></div>
      <div class="sg-item"><span class="sg-label">Expectancy (R)</span><span class="${pnlClass(stats.expectancy)}">${stats.expectancy === null || stats.expectancy === undefined ? "--" : fmtNum(stats.expectancy, 2)}</span></div>
    </div>
  `;
}

function baseChartLayout() {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 55, r: 20, t: 10, b: 40 },
    font: { color: "#c9d1d9", family: "JetBrains Mono, monospace", size: 11 },
    xaxis: { gridcolor: "#232a37", showgrid: true, zeroline: false },
    yaxis: { gridcolor: "#232a37", showgrid: true, zeroline: false, title: "Equity" },
    legend: { orientation: "h", y: -0.2 },
    showlegend: true,
    hovermode: "x unified",
  };
}

function plotlyConfig() {
  return { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
}

function renderDetailChart(curve) {
  const isPct = state.pnlMode === "percent";
  const display = equityToDisplay(curve);
  const trace = {
    x: display.map((p) => p.t),
    y: display.map((p) => p.e),
    type: "scattergl",
    mode: "lines",
    name: "equity",
    line: { color: "#58a6ff", width: 1.8 },
    fill: "tozeroy",
    fillcolor: "rgba(88, 166, 255, 0.06)",
  };
  const layout = baseChartLayout();
  layout.yaxis = { ...layout.yaxis, title: isPct ? "Return %" : "Equity" };
  layout.showlegend = false;

  if (!state.detailChartInitialized) {
    Plotly.newPlot(el.detailChart, [trace], layout, plotlyConfig());
    state.detailChartInitialized = true;
  } else {
    Plotly.react(el.detailChart, [trace], layout, plotlyConfig());
  }
}

function renderDrawdownChart(curve) {
  const isPct = state.pnlMode === "percent";
  let peak = -Infinity;
  const dd = curve.map((p) => {
    peak = Math.max(peak, p.e);
    if (isPct) return peak > 0 ? ((p.e - peak) / peak) * 100 : 0;
    return p.e - peak;
  });
  const trace = {
    x: curve.map((p) => p.t),
    y: dd,
    type: "scattergl",
    mode: "lines",
    line: { color: "#f85149", width: 1.5 },
    fill: "tozeroy",
    fillcolor: "rgba(248, 81, 73, 0.15)",
  };
  const layout = baseChartLayout();
  layout.yaxis = { ...layout.yaxis, title: isPct ? "Drawdown %" : "Drawdown" };
  layout.showlegend = false;

  if (!state.drawdownChartInitialized) {
    Plotly.newPlot(el.drawdownChart, [trace], layout, plotlyConfig());
    state.drawdownChartInitialized = true;
  } else {
    Plotly.react(el.drawdownChart, [trace], layout, plotlyConfig());
  }
}

// GitHub-style daily PnL calendar. Colored by magnitude relative to the
// worst/best day in the received window (no fixed $ thresholds — strategies
// run at different capital/position sizes).
function heatmapCellClass(pnl, maxAbs) {
  if (!pnl || !maxAbs || maxAbs <= 0) return "";
  const tier = Math.min(3, Math.max(1, Math.ceil((Math.abs(pnl) / maxAbs) * 3)));
  return pnl > 0 ? `hm-win-${tier}` : `hm-loss-${tier}`;
}

function renderCalendarHeatmap(dailyPnl) {
  const WEEKS = 14;
  const today = new Date();
  today.setUTCHours(0, 0, 0, 0);
  const start = new Date(today);
  start.setUTCDate(start.getUTCDate() - (WEEKS * 7 - 1));
  start.setUTCDate(start.getUTCDate() - ((start.getUTCDay() + 6) % 7)); // roll back to Monday

  const base = state.detailStartingEquity;
  const display = {};
  for (const [d, v] of Object.entries(dailyPnl)) display[d] = amountToDisplay(v, base);
  const maxAbs = Math.max(0, ...Object.values(display).map((v) => Math.abs(v)));

  const totalDays = Math.round((today - start) / 86400000) + 1;
  const totalCols = Math.ceil(totalDays / 7);

  const cellsHtml = [];
  const monthMarks = []; // { col, label } — one per calendar-month transition, GitHub style
  let lastMonthKey = null;
  const cursor = new Date(start);
  for (let i = 0; i < totalDays; i++) {
    const dow = (cursor.getUTCDay() + 6) % 7; // 0=Mon..6=Sun
    const col = Math.floor(i / 7);
    if (dow === 0) {
      const monthKey = `${cursor.getUTCFullYear()}-${cursor.getUTCMonth()}`;
      if (monthKey !== lastMonthKey) {
        monthMarks.push({ col, label: cursor.toLocaleString("en-US", { month: "short", timeZone: "UTC" }) });
        lastMonthKey = monthKey;
      }
    }
    const iso = cursor.toISOString().slice(0, 10);
    const pnl = Object.prototype.hasOwnProperty.call(display, iso) ? display[iso] : null;
    const cls = pnl === null ? "" : heatmapCellClass(pnl, maxAbs);
    // Full weekday + date in the tooltip, not just ISO — "no hover needed to
    // tell the day/month" is the visible month/weekday labels below; this is
    // the exact-date detail on top of that.
    const dateLabel = cursor.toLocaleString("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });
    const title = pnl === null ? `${dateLabel}: no closed trades` : `${dateLabel}: ${fmtDisplayValue(pnl)}`;
    cellsHtml.push(`<div class="heatmap-cell ${cls}" title="${escapeHtml(title)}"></div>`);
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }

  const colStyle = `grid-template-columns: repeat(${totalCols}, 15px);`;
  const monthsHtml = monthMarks
    .map((m) => `<span style="grid-column:${m.col + 1}">${m.label}</span>`)
    .join("");

  el.calendarHeatmap.innerHTML = `
    <div class="heatmap-wrap">
      <div class="heatmap-weekday-col">
        <span></span><span>Mon</span><span></span><span>Wed</span><span></span><span>Fri</span><span></span>
      </div>
      <div class="heatmap-main">
        <div class="heatmap-months" style="${colStyle}">${monthsHtml}</div>
        <div class="heatmap-grid" style="${colStyle}">${cellsHtml.join("")}</div>
      </div>
    </div>
    <div class="heatmap-legend">
      <span>loss</span>
      <span class="heatmap-cell hm-loss-3"></span><span class="heatmap-cell hm-loss-2"></span><span class="heatmap-cell hm-loss-1"></span>
      <span class="heatmap-cell"></span>
      <span class="heatmap-cell hm-win-1"></span><span class="heatmap-cell hm-win-2"></span><span class="heatmap-cell hm-win-3"></span>
      <span>win</span>
    </div>
  `;
}

function renderRMultipleHistogram(rMultiples) {
  const wins = rMultiples.filter((d) => d.r >= 0).map((d) => d.r);
  const losses = rMultiples.filter((d) => d.r < 0).map((d) => d.r);
  const traces = [
    { x: wins, type: "histogram", name: "R ≥ 0", marker: { color: "#3fb950" }, opacity: 0.85 },
    { x: losses, type: "histogram", name: "R < 0", marker: { color: "#f85149" }, opacity: 0.85 },
  ];
  const layout = baseChartLayout();
  layout.yaxis = { ...layout.yaxis, title: "Count" };
  layout.xaxis = { ...layout.xaxis, title: "R multiple" };
  layout.barmode = "overlay";
  layout.showlegend = false;

  if (!state.rMultipleChartInitialized) {
    Plotly.newPlot(el.rMultipleChart, traces, layout, plotlyConfig());
    state.rMultipleChartInitialized = true;
  } else {
    Plotly.react(el.rMultipleChart, traces, layout, plotlyConfig());
  }
}

function renderSessionStats(sessionStats) {
  const isPct = state.pnlMode === "percent";
  const base = state.detailStartingEquity;
  const values = sessionStats.map((s) => amountToDisplay(s.total_pnl, base));
  const traces = [{
    x: sessionStats.map((s) => s.session),
    y: values,
    type: "bar",
    marker: { color: values.map((v) => (v >= 0 ? "#3fb950" : "#f85149")) },
    text: sessionStats.map((s) => `${fmtInt(s.trades)} trades · ${fmtPct(s.win_rate)} win`),
    hoverinfo: "text+y",
  }];
  const layout = baseChartLayout();
  layout.yaxis = { ...layout.yaxis, title: isPct ? "Total Return %" : "Total PnL" };
  layout.showlegend = false;

  if (!state.sessionChartInitialized) {
    Plotly.newPlot(el.sessionChart, traces, layout, plotlyConfig());
    state.sessionChartInitialized = true;
  } else {
    Plotly.react(el.sessionChart, traces, layout, plotlyConfig());
  }
}

// Plain-language fallback for rows where the backend didn't write a `note`
// (older log entries from before that field existed, or ENTRY/EXIT/PARTIAL
// rows where the event name mostly speaks for itself) — the goal is that
// *no* row ever shows a blank reason.
function fallbackReason(t, dir) {
  const qty = t.qty === null || t.qty === undefined ? null : fmtNum(t.qty, 4);
  const pnl = t.pnl_net === null || t.pnl_net === undefined ? null : signedFmt(t.pnl_net);
  if (t.event === "ENTRY") {
    return `Opened ${dir.text} ${qty ?? ""} @ ${fmtNum(t.price)}`.trim();
  }
  if (t.event && t.event.startsWith("EXIT_")) {
    return `Closed @ ${fmtNum(t.price)}${pnl ? ` — ${pnl} net` : ""}`;
  }
  if (t.event && t.event.startsWith("PARTIAL_")) {
    return `Partial close ${qty ?? ""} @ ${fmtNum(t.price)}${pnl ? ` — ${pnl} net` : ""}`.trim();
  }
  return "no reason recorded";
}

// Signals list: SIGNAL rows only ("no trade, here's why") — actual trades
// live in the separate Trades table (renderTradesTable) below.
function renderSignalsList(trades) {
  const signals = trades.filter((t) => t.event === "SIGNAL");
  if (!signals.length) {
    el.signalsListBody.innerHTML = `<div class="empty-row">No signals yet.</div>`;
    return;
  }
  const rows = signals.map((t) => {
    const dir = dirLabel(t.direction);
    const evt = eventInfo(t.event, t.direction, t.pnl_net);
    const reasonText = escapeHtml(t.note || fallbackReason(t, dir));
    return `
      <div class="trade-row ${evt.row}">
        <div class="trade-row-accent ${evt.cls}"></div>
        <div class="trade-row-body">
          <div class="trade-row-main">
            <span class="trade-time">${fmtTime(t.timestamp)}</span>
            <span class="evt-badge ${evt.cls}">${evt.text}</span>
            <span class="${dir.cls}">${dir.text}</span>
            <span class="trade-price">${fmtNum(t.price)}</span>
          </div>
          <div class="trade-reason">${reasonText}</div>
        </div>
      </div>
    `;
  });
  el.signalsListBody.innerHTML = rows.join("");
}

// One row per trade episode (ENTRY plus everything it closed with) — merges
// state.tradeEpisodes (closed/partially-closed history) with the currently
// open position when it has zero closes yet (so a brand-new entry shows up
// immediately, not just after its first partial/exit — see resolveTradeTarget
// for why position.json's entry_time needs isSameEntry, not exact match).
function buildTradeRows() {
  const rows = state.tradeEpisodes.map((ep) => {
    const isCurrent = state.currentPosition && isSameEntry(state.currentPosition.entry_time, ep.entry_time);
    return isCurrent ? normalizeTarget(true, state.currentPosition, ep) : normalizeTarget(false, null, ep);
  });
  if (state.currentPosition) {
    const matched = state.tradeEpisodes.some((ep) => isSameEntry(state.currentPosition.entry_time, ep.entry_time));
    if (!matched) rows.push(normalizeTarget(true, state.currentPosition, null));
  }
  rows.sort((a, b) => (b.entryTime || "").localeCompare(a.entryTime || ""));
  return rows;
}

function exitReasonLabel(t, lastClose) {
  if (t.isOpen) return lastClose ? "partial (open)" : "open";
  if (!lastClose) return "--";
  return String(lastClose.event || "").replace("EXIT_", "").replace(/_/g, " ");
}

function renderTradesTable(rows) {
  if (!rows.length) {
    el.tradesTableBody.innerHTML = `<tr><td colspan="16" class="empty-row">No trades yet.</td></tr>`;
    return;
  }
  const html = rows.map((t) => {
    const dir = dirLabel(t.direction);
    const lastClose = t.closes.length ? t.closes[t.closes.length - 1] : null;
    const closePrice = t.isOpen ? currentMarkPrice() : (lastClose ? lastClose.price : null);
    const invested = (t.entryPrice || 0) * (t.qtyTotal || 0);
    const pctOfEquity = t.equityAtEntry ? (invested / t.equityAtEntry) * 100 : null;
    const valueAtClose = (closePrice !== null && closePrice !== undefined && t.qtyTotal !== null)
      ? closePrice * t.qtyTotal : null;

    const realizedSoFar = t.closes.reduce((s, c) => s + c.pnl_net, 0);
    const closedQty = t.closes.reduce((s, c) => s + (c.qty || 0), 0);
    const remainingQty = Math.max(0, (t.qtyTotal || 0) - closedQty);
    const floating = (t.isOpen && closePrice !== null && closePrice !== undefined)
      ? t.direction * (closePrice - t.entryPrice) * remainingQty : 0;
    const pnlAbs = realizedSoFar + floating;
    const pnlPct = invested ? (pnlAbs / invested) * 100 : null;

    const durationMs = (t.isOpen ? Date.now() : (lastClose ? new Date(lastClose.time).getTime() : null))
      - new Date(t.entryTime).getTime();
    const feesTotal = t.closes.reduce((s, c) => s + (c.fee || 0), 0) || t.feesTotal;

    // "Close" is only the LAST leg's price — for a partial+final trade most
    // of the pnl can come from an earlier leg at a very different price
    // (e.g. a TP1 partial), so a lone close price makes the pnl look
    // unexplained. Flag it and let hover show every leg's own price/qty/pnl.
    const legsBadge = t.closes.length > 1
      ? ` <span class="legs-badge" title="${escapeHtml(t.closes.map((c) =>
          `${String(c.event || "").replace(/_/g, " ")}: ${fmtNum(c.qty, 4)} @ ${fmtNum(c.price)} (${signedFmt(c.pnl_net)})`
        ).join(" → "))}">×${t.closes.length}</span>`
      : "";

    // Partial-close price(s), shown plainly (not just on hover) since most
    // of a trade's pnl often comes from an earlier partial at a very
    // different price than the final "Close" column.
    const partialPrices = t.closes.filter((c) => String(c.event || "").startsWith("PARTIAL"));
    const partialCell = partialPrices.length
      ? partialPrices.map((c) => fmtNum(c.price)).join(", ")
      : "--";

    return `
      <tr class="clickable" data-entry-time="${escapeHtml(t.entryTime)}">
        <td class="${dir.cls}">${dir.text}</td>
        <td>${fmtNum(t.entryPrice)}</td>
        <td>${fmtNum(t.sl)}</td>
        <td>${fmtNum(t.tp1)}</td>
        <td>${partialCell}</td>
        <td>${t.isOpen ? "open" : fmtNum(closePrice)}${legsBadge}</td>
        <td>${escapeHtml(exitReasonLabel(t, lastClose))}</td>
        <td>${durationMs > 0 ? fmtUptime(durationMs / 1000) : "--"}</td>
        <td>${fmtNum(t.qtyTotal, 4)}</td>
        <td>${fmtNum(invested)}</td>
        <td>${pctOfEquity === null ? "--" : fmtNum(pctOfEquity, 1) + "%"}</td>
        <td>${valueAtClose === null ? "--" : fmtNum(valueAtClose)}</td>
        <td>${feesTotal ? fmtNum(feesTotal) : "--"}</td>
        <td class="${pnlClass(pnlAbs)}">${signedFmt(pnlAbs)}</td>
        <td class="${pnlClass(pnlPct)}">${pnlPct === null ? "--" : `${pnlPct >= 0 ? "+" : ""}${pnlPct.toFixed(2)}%`}</td>
        <td class="${pnlClass(t.rMultiple)}">${t.rMultiple === null || t.rMultiple === undefined ? "--" : `${t.rMultiple >= 0 ? "+" : ""}${t.rMultiple.toFixed(2)}R`}</td>
      </tr>
    `;
  });
  el.tradesTableBody.innerHTML = html.join("");

  el.tradesTableBody.querySelectorAll("tr.clickable").forEach((tr) => {
    const target = resolveTradeTarget(tr.dataset.entryTime);
    if (target) tr.addEventListener("click", () => openTradeModal(target));
  });
}

function classifyLogLine(line) {
  if (line.includes(" ERROR ") || line.includes("CRITICAL")) return "log-error";
  if (line.includes(" WARNING ")) return "log-warn";
  return "";
}

function renderLogTail(lines) {
  const prev = state.lastLogLines;

  const atBottom =
    el.logTail.scrollTop + el.logTail.clientHeight >= el.logTail.scrollHeight - 10;

  if (prev.length === 0 || el.logTail.childElementCount === 0) {
    // Full (re)build.
    const frag = document.createDocumentFragment();
    for (const line of lines) {
      frag.appendChild(makeLogLineEl(line));
    }
    el.logTail.innerHTML = "";
    el.logTail.appendChild(frag);
  } else {
    const lastPrevLine = prev[prev.length - 1];
    let idx = -1;
    for (let i = lines.length - 1; i >= 0; i--) {
      if (lines[i] === lastPrevLine) { idx = i; break; }
    }
    if (idx === -1) {
      // Buffer rotated/truncated beyond what we can diff — rebuild fully.
      const frag = document.createDocumentFragment();
      for (const line of lines) {
        frag.appendChild(makeLogLineEl(line));
      }
      el.logTail.innerHTML = "";
      el.logTail.appendChild(frag);
    } else if (idx < lines.length - 1) {
      const frag = document.createDocumentFragment();
      for (let i = idx + 1; i < lines.length; i++) {
        frag.appendChild(makeLogLineEl(lines[i]));
      }
      el.logTail.appendChild(frag);
      // Trim excess DOM nodes from the top if the list grew beyond backend cap.
      while (el.logTail.childElementCount > 200) {
        el.logTail.removeChild(el.logTail.firstChild);
      }
    }
    // else: no new lines, nothing to do.
  }

  state.lastLogLines = lines;

  if (atBottom) {
    el.logTail.scrollTop = el.logTail.scrollHeight;
  }
}

function makeLogLineEl(line) {
  const div = document.createElement("div");
  const cls = classifyLogLine(line);
  div.className = "log-line" + (cls ? " " + cls : "");
  div.textContent = line;
  return div;
}

// ---------------------------------------------------------------
// Live BTCUSDT market chart (TradingView Lightweight Charts)
// Talks directly to Binance Futures public REST/WS endpoints —
// deliberately NOT proxied through our own backend.
// ---------------------------------------------------------------

const BINANCE_REST = "https://fapi.binance.com/fapi/v1/klines";
const BINANCE_WS = "wss://fstream.binance.com/ws";
const market = {
  chart: null,
  candleSeries: null,
  volumeSeries: null,
  socket: null,
  timeframe: "1h",
  resizeObserver: null,
  lastCandle: null,
  prevClose: null,
};

const tradeModal = { chart: null, candleSeries: null };

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setMarketConnState(connState) {
  el.marketConn.classList.remove("reconnecting", "down");
  if (connState === "open") {
    el.marketConnLabel.textContent = "live";
  } else if (connState === "reconnecting") {
    el.marketConn.classList.add("reconnecting");
    el.marketConnLabel.textContent = "reconnecting…";
  } else if (connState === "connecting") {
    el.marketConn.classList.add("reconnecting");
    el.marketConnLabel.textContent = "connecting…";
  } else {
    el.marketConn.classList.add("down");
    el.marketConnLabel.textContent = "disconnected";
  }
}

function parseKline(k) {
  // Binance REST kline array: [openTime, open, high, low, close, volume, closeTime, ...]
  return {
    time: Math.floor(k[0] / 1000),
    open: Number(k[1]),
    high: Number(k[2]),
    low: Number(k[3]),
    close: Number(k[4]),
    volume: Number(k[5]),
  };
}

function volumeBar(c) {
  const up = c.close >= c.open;
  return {
    time: c.time,
    value: c.volume,
    color: up ? "rgba(63, 185, 80, 0.5)" : "rgba(248, 81, 73, 0.5)",
  };
}

function initMarketChart() {
  if (typeof LightweightCharts === "undefined") return; // CDN failed to load; degrade gracefully
  if (!el.marketChart) return;

  const textColor = cssVar("--text") || "#c9d1d9";
  const gridColor = cssVar("--border") || "#232a37";
  const dimColor = cssVar("--text-dim") || "#7d8590";
  const green = cssVar("--green") || "#3fb950";
  const red = cssVar("--red") || "#f85149";

  market.chart = LightweightCharts.createChart(el.marketChart, {
    layout: {
      background: { type: "solid", color: "transparent" },
      textColor,
      fontFamily: "JetBrains Mono, monospace",
      fontSize: 11,
    },
    grid: {
      vertLines: { color: gridColor },
      horzLines: { color: gridColor },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: gridColor },
    timeScale: { borderColor: gridColor, timeVisible: true, secondsVisible: false },
    watermark: { visible: false },
    autoSize: false,
    width: el.marketChart.clientWidth,
    height: el.marketChart.clientHeight || 460,
  });

  market.candleSeries = market.chart.addCandlestickSeries({
    upColor: green,
    downColor: red,
    borderUpColor: green,
    borderDownColor: red,
    wickUpColor: green,
    wickDownColor: red,
    priceScaleId: "right",
  });
  market.candleSeries.priceScale().applyOptions({
    scaleMargins: { top: 0.06, bottom: 0.28 },
  });

  market.volumeSeries = market.chart.addHistogramSeries({
    priceFormat: { type: "volume" },
    priceScaleId: "vol",
    color: dimColor,
  });
  market.chart.priceScale("vol").applyOptions({
    scaleMargins: { top: 0.78, bottom: 0 },
  });

  // Responsive sizing.
  market.resizeObserver = new ResizeObserver(() => {
    if (!market.chart) return;
    market.chart.applyOptions({
      width: el.marketChart.clientWidth,
      height: el.marketChart.clientHeight || 460,
    });
  });
  market.resizeObserver.observe(el.marketChart);
  window.addEventListener("resize", () => {
    if (!market.chart) return;
    market.chart.applyOptions({
      width: el.marketChart.clientWidth,
      height: el.marketChart.clientHeight || 460,
    });
  });

  el.tfSelector.addEventListener("click", (evt) => {
    const btn = evt.target.closest(".tf-btn");
    if (!btn) return;
    const tf = btn.dataset.tf;
    if (tf === market.timeframe) return;
    el.tfSelector.querySelectorAll(".tf-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    changeTimeframe(tf);
  });

  loadMarketHistory(market.timeframe).then(() => setupMarketSocket(market.timeframe));
}

async function loadMarketHistory(interval) {
  try {
    const resp = await fetch(`${BINANCE_REST}?symbol=BTCUSDT&interval=${interval}&limit=500`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const raw = await resp.json();
    const candles = raw.map(parseKline);
    if (!candles.length) return;

    market.candleSeries.setData(candles.map((c) => ({
      time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
    })));
    market.volumeSeries.setData(candles.map(volumeBar));

    const last = candles[candles.length - 1];
    market.lastCandle = last;
    market.prevClose = candles.length > 1 ? candles[candles.length - 2].close : last.open;
    updateMarketPriceDisplay(last.close);
    market.chart.timeScale().fitContent();
  } catch (e) {
    setMarketConnState("closed");
  }
}

function updateMarketPriceDisplay(price) {
  el.marketLastPrice.textContent = fmtNum(price, 1);
  if (market.prevClose === null || market.prevClose === undefined) {
    el.marketLastChange.textContent = "--";
    el.marketLastChange.className = "market-last-change";
    return;
  }
  const diff = price - market.prevClose;
  const pct = market.prevClose !== 0 ? (diff / market.prevClose) * 100 : 0;
  el.marketLastChange.textContent = `${diff >= 0 ? "+" : ""}${fmtNum(diff, 1)} (${diff >= 0 ? "+" : ""}${pct.toFixed(2)}%)`;
  el.marketLastChange.className = "market-last-change " + pnlClass(diff);
}

function setupMarketSocket(interval) {
  const url = `${BINANCE_WS}/btcusdt@kline_${interval}`;
  market.socket = new ReconnectingSocket(url, handleMarketMessage, setMarketConnState);
}

function handleMarketMessage(msg) {
  if (!msg || msg.e !== "kline" || !msg.k) return;
  const k = msg.k;
  const candle = {
    time: Math.floor(k.t / 1000),
    open: Number(k.o),
    high: Number(k.h),
    low: Number(k.l),
    close: Number(k.c),
    volume: Number(k.v),
  };

  market.candleSeries.update({
    time: candle.time, open: candle.open, high: candle.high, low: candle.low, close: candle.close,
  });
  market.volumeSeries.update(volumeBar(candle));
  updateMarketPriceDisplay(candle.close);

  if (k.x) {
    // Candle closed: shift prevClose reference forward for the next one.
    market.prevClose = candle.close;
  }
  market.lastCandle = candle;
  refreshUnrealizedPnl();
}

function changeTimeframe(tf) {
  market.timeframe = tf;
  if (market.socket) {
    market.socket.close();
    market.socket = null;
  }
  setMarketConnState("connecting");
  loadMarketHistory(tf).then(() => setupMarketSocket(tf));
}

// ---------------------------------------------------------------
// Trade replay modal
// ---------------------------------------------------------------

// position.json's entry_time and trades.csv's ENTRY timestamp are two
// independent datetime.now() calls a few ms apart in the trader — never
// byte-identical — so matching "is this the currently open position" needs
// a tolerance instead of string equality.
function isSameEntry(tsA, tsB, toleranceMs = 5000) {
  if (!tsA || !tsB) return false;
  return Math.abs(new Date(tsA).getTime() - new Date(tsB).getTime()) < toleranceMs;
}

function normalizeTarget(fromPosition, position, episode) {
  if (fromPosition) {
    return {
      isOpen: true,
      entryTime: episode ? episode.entry_time : position.entry_time,
      direction: position.direction,
      entryPrice: position.entry_price,
      sl: position.sl,
      tp1: position.tp1,
      tp2: position.tp2,
      tp3: position.tp3,
      qtyTotal: position.size_total,
      equityAtEntry: episode ? episode.equity_at_entry : position.equity_at_entry,
      feesTotal: episode ? episode.fees_total : null,
      rMultiple: episode ? episode.r_multiple : null,
      closes: episode ? episode.closes : [],
    };
  }
  return {
    isOpen: false,
    entryTime: episode.entry_time,
    direction: episode.direction,
    entryPrice: episode.entry_price,
    sl: episode.sl,
    tp1: episode.tp1,
    tp2: episode.tp2,
    tp3: episode.tp3,
    qtyTotal: episode.qty_total,
    equityAtEntry: episode.equity_at_entry,
    feesTotal: episode.fees_total,
    rMultiple: episode.r_multiple,
    closes: episode.closes,
  };
}

function resolveTradeTarget(entryTime) {
  if (!entryTime) return null;
  const episode = state.tradeEpisodes.find((e) => e.entry_time === entryTime);
  const isCurrentPosition = state.currentPosition
    && isSameEntry(state.currentPosition.entry_time, entryTime);
  if (isCurrentPosition) return normalizeTarget(true, state.currentPosition, episode || null);
  return episode ? normalizeTarget(false, null, episode) : null;
}

const REPLAY_INTERVAL_MS = { "1m": 60e3, "5m": 300e3, "15m": 900e3, "1h": 3600e3, "4h": 14400e3 };

function pickReplayInterval(durationMs) {
  const hours = durationMs / 3600e3;
  if (hours <= 3) return "1m";
  if (hours <= 12) return "5m";
  if (hours <= 48) return "15m";
  if (hours <= 240) return "1h";
  return "4h";
}

// Combined realized-so-far + floating PnL at each candle (not floating-only,
// which would jump discontinuously at every partial close) and its
// running-peak drawdown — the "how did this trade's value evolve" pair.
function computeTradeSeries(candles, target) {
  const entryMs = new Date(target.entryTime).getTime();
  const closes = target.closes
    .map((c) => ({ ...c, ms: new Date(c.time).getTime() }))
    .sort((a, b) => a.ms - b.ms);
  const lastCloseMs = closes.length ? closes[closes.length - 1].ms : null;

  let peak = -Infinity;
  const pnlPoints = [];
  const ddPoints = [];
  for (const c of candles) {
    const cMs = c.time * 1000;
    if (cMs < entryMs) continue;
    if (!target.isOpen && lastCloseMs !== null && cMs > lastCloseMs) break;

    let closedQty = 0;
    let realizedSoFar = 0;
    for (const cl of closes) {
      if (cl.ms <= cMs) {
        closedQty += cl.qty || 0;
        realizedSoFar += cl.pnl_net;
      }
    }
    const remaining = Math.max(0, (target.qtyTotal || 0) - closedQty);
    const floating = target.direction * (c.close - target.entryPrice) * remaining;
    const total = floating + realizedSoFar;

    pnlPoints.push({ t: c.time, v: total });
    peak = Math.max(peak, total);
    ddPoints.push({ t: c.time, v: total - peak });
  }
  return { pnlPoints, ddPoints };
}

function renderTradeModalChart(candles, target) {
  if (tradeModal.chart) {
    tradeModal.chart.remove();
    tradeModal.chart = null;
    tradeModal.candleSeries = null;
  }
  if (typeof LightweightCharts === "undefined" || !el.tradeModalChart) return;

  const textColor = cssVar("--text") || "#c9d1d9";
  const gridColor = cssVar("--border") || "#232a37";
  const green = cssVar("--green") || "#3fb950";
  const red = cssVar("--red") || "#f85149";

  tradeModal.chart = LightweightCharts.createChart(el.tradeModalChart, {
    layout: { background: { type: "solid", color: "transparent" }, textColor, fontFamily: "JetBrains Mono, monospace", fontSize: 11 },
    grid: { vertLines: { color: gridColor }, horzLines: { color: gridColor } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: gridColor },
    timeScale: { borderColor: gridColor, timeVisible: true, secondsVisible: false },
    watermark: { visible: false },
    autoSize: false,
    width: el.tradeModalChart.clientWidth,
    height: el.tradeModalChart.clientHeight || 380,
  });

  tradeModal.candleSeries = tradeModal.chart.addCandlestickSeries({
    upColor: green, downColor: red, borderUpColor: green, borderDownColor: red,
    wickUpColor: green, wickDownColor: red,
  });
  tradeModal.candleSeries.setData(candles.map((c) => (
    { time: c.time, open: c.open, high: c.high, low: c.low, close: c.close }
  )));

  const markers = [{
    time: Math.floor(new Date(target.entryTime).getTime() / 1000),
    position: target.direction === 1 ? "belowBar" : "aboveBar",
    color: target.direction === 1 ? green : red,
    shape: target.direction === 1 ? "arrowUp" : "arrowDown",
    text: "ENTRY",
  }];
  for (const c of target.closes) {
    markers.push({
      time: Math.floor(new Date(c.time).getTime() / 1000),
      position: "aboveBar",
      color: c.pnl_net >= 0 ? green : red,
      shape: "circle",
      text: String(c.event || "").replace("EXIT_", "").replace("PARTIAL_", "P:").replace(/_/g, " "),
    });
  }
  markers.sort((a, b) => a.time - b.time);
  tradeModal.candleSeries.setMarkers(markers);

  const priceLine = (price, color, title) => {
    if (price === null || price === undefined) return;
    tradeModal.candleSeries.createPriceLine({
      price, color, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true, title,
    });
  };
  priceLine(target.sl, red, "SL");
  priceLine(target.tp1, green, "TP1");
  priceLine(target.tp2, green, "TP2");
  priceLine(target.tp3, green, "TP3");

  tradeModal.chart.timeScale().fitContent();
}

function renderTradeModalPnl(pnlPoints, notional) {
  const isPct = state.pnlMode === "percent" && notional;
  const trace = {
    x: pnlPoints.map((p) => new Date(p.t * 1000).toISOString()),
    y: pnlPoints.map((p) => (isPct ? (p.v / notional) * 100 : p.v)),
    type: "scattergl",
    mode: "lines",
    line: { color: "#58a6ff", width: 1.6 },
    fill: "tozeroy",
    fillcolor: "rgba(88, 166, 255, 0.08)",
  };
  const layout = baseChartLayout();
  layout.yaxis = { ...layout.yaxis, title: isPct ? "PnL % (of notional)" : "PnL" };
  layout.showlegend = false;
  Plotly.newPlot(el.tradeModalPnl, [trace], layout, plotlyConfig());
}

function renderTradeModalDrawdown(ddPoints, notional) {
  const isPct = state.pnlMode === "percent" && notional;
  const trace = {
    x: ddPoints.map((p) => new Date(p.t * 1000).toISOString()),
    y: ddPoints.map((p) => (isPct ? (p.v / notional) * 100 : p.v)),
    type: "scattergl",
    mode: "lines",
    line: { color: "#f85149", width: 1.4 },
    fill: "tozeroy",
    fillcolor: "rgba(248, 81, 73, 0.15)",
  };
  const layout = baseChartLayout();
  layout.yaxis = { ...layout.yaxis, title: isPct ? "Drawdown % (of notional)" : "Drawdown" };
  layout.showlegend = false;
  Plotly.newPlot(el.tradeModalDrawdown, [trace], layout, plotlyConfig());
}

async function openTradeModal(target) {
  el.tradeModal.classList.remove("hidden");

  const dir = dirLabel(target.direction);
  const lastClose = target.closes[target.closes.length - 1];
  const exitLabel = target.isOpen ? "open" : fmtNum(lastClose ? lastClose.price : null);
  el.tradeModalTitle.textContent = `${dir.text} · ${fmtNum(target.entryPrice)} → ${exitLabel}`;
  el.tradeModalSubtitle.textContent = `Entry ${fmtTime(target.entryTime)}`
    + (target.isOpen ? " — still open" : ` · pnl ${signedFmt(target.closes.reduce((s, c) => s + c.pnl_net, 0))}`);

  const entryMs = new Date(target.entryTime).getTime();
  const endMs = target.isOpen ? Date.now() : new Date(lastClose.time).getTime();
  const durationMs = Math.max(endMs - entryMs, 60000);
  const interval = pickReplayInterval(durationMs);
  const padMs = Math.max(durationMs * 0.15, (REPLAY_INTERVAL_MS[interval] || 60000) * 10);
  const fetchStart = Math.floor(entryMs - padMs);
  const fetchEnd = Math.min(Math.floor(endMs + padMs), Date.now());

  let candles = [];
  try {
    const resp = await fetch(`${BINANCE_REST}?symbol=BTCUSDT&interval=${interval}&startTime=${fetchStart}&endTime=${fetchEnd}&limit=1000`);
    if (resp.ok) candles = (await resp.json()).map(parseKline);
  } catch (e) {
    // leave candles empty — the modal still shows PnL/drawdown computed
    // from the trade's own recorded prices even if Binance is unreachable.
  }

  renderTradeModalChart(candles, target);
  const { pnlPoints, ddPoints } = computeTradeSeries(candles, target);
  const notional = (target.entryPrice || 0) * (target.qtyTotal || 0) || null;
  renderTradeModalPnl(pnlPoints, notional);
  renderTradeModalDrawdown(ddPoints, notional);
}

function closeTradeModal() {
  el.tradeModal.classList.add("hidden");
  if (tradeModal.chart) {
    tradeModal.chart.remove();
    tradeModal.chart = null;
    tradeModal.candleSeries = null;
  }
}

el.tradeModalClose.addEventListener("click", closeTradeModal);
el.tradeModal.addEventListener("click", (e) => {
  if (e.target === el.tradeModal) closeTradeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !el.tradeModal.classList.contains("hidden")) closeTradeModal();
});

// ---------------------------------------------------------------
// Boot
// ---------------------------------------------------------------

const overviewSocket = new ReconnectingSocket(
  `${wsBase}/ws/overview`,
  handleOverviewMessage,
  setConnState
);

// Keep the clock ticking between messages for a livelier feel.
setInterval(() => {
  if (el.serverClock.textContent === "--:--:--") return;
}, 1000);

// Keep the "Latest Analysis" relative timestamp live between WS ticks.
setInterval(() => {
  if (state.currentView !== "overview" && state.lastAnalysisTimestamp) {
    el.analysisAge.textContent = fmtRelativeAge(state.lastAnalysisTimestamp);
  }
}, 1000);

initMarketChart();
