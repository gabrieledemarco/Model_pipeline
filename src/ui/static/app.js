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

  detailError: document.getElementById("detail-error"),
  detailHeader: document.getElementById("detail-header"),
  detailTitle: document.getElementById("detail-title"),
  detailBadges: document.getElementById("detail-badges"),
  detailUptime: document.getElementById("detail-uptime"),
  detailCpu: document.getElementById("detail-cpu"),
  detailRss: document.getElementById("detail-rss"),
  positionBody: document.getElementById("position-body"),
  statsBody: document.getElementById("stats-body"),
  detailChart: document.getElementById("detail-chart"),
  tradesTableBody: document.getElementById("trades-table-body"),
  logTail: document.getElementById("log-tail"),
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
  detailSocket: null,
  lastLogLines: [],
  lastDetailStrategyId: null,
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
    return `
      <tr class="clickable" data-id="${escapeHtml(s.strategy_id)}">
        <td class="mono">${escapeHtml(s.strategy_id)}</td>
        <td>${escapeHtml(s.scenario || "")}</td>
        <td>${escapeHtml(s.exchange || "")}</td>
        <td>${dryBadge}</td>
        <td><span class="badge badge-${info.cls}">${info.label}</span></td>
        <td class="${posCls}">${escapeHtml(s.position_desc || "flat")}</td>
        <td class="${pnlClass(s.realized_pnl)}">${signedFmt(s.realized_pnl)}</td>
        <td>${fmtNum(s.equity)}</td>
      </tr>
    `;
  });
  el.overviewTableBody.innerHTML = rows.length
    ? rows.join("")
    : `<tr><td colspan="8" class="empty-row">No strategies reported.</td></tr>`;

  el.overviewTableBody.querySelectorAll("tr.clickable").forEach((tr) => {
    tr.addEventListener("click", () => selectStrategy(tr.dataset.id));
  });

  renderOverviewChart(agg.equity_curves || {});
}

function renderOverviewChart(equityCurves) {
  const ids = Object.keys(equityCurves);
  const traces = ids.map((id) => {
    const pts = equityCurves[id] || [];
    return {
      x: pts.map((p) => p.t),
      y: pts.map((p) => p.e),
      type: "scattergl",
      mode: "lines",
      name: id,
      line: { color: colorFor(id), width: 1.6 },
      hovertemplate: "%{y:.2f}<br>%{x}<extra>" + id + "</extra>",
    };
  });

  const layout = baseChartLayout();

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
  const meta = msg.meta || {};
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

  renderPosition(msg.position || { active: false });
  renderStats(msg.stats || { insufficient_data: true });
  renderDetailChart(msg.equity_curve || []);
  renderTrades(msg.trades || []);
  renderLogTail(msg.log_tail || []);
}

function renderPosition(pos) {
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
    </div>
  `;
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
  const trace = {
    x: curve.map((p) => p.t),
    y: curve.map((p) => p.e),
    type: "scattergl",
    mode: "lines",
    name: "equity",
    line: { color: "#58a6ff", width: 1.8 },
    fill: "tozeroy",
    fillcolor: "rgba(88, 166, 255, 0.06)",
  };
  const layout = baseChartLayout();
  layout.showlegend = false;

  if (!state.detailChartInitialized) {
    Plotly.newPlot(el.detailChart, [trace], layout, plotlyConfig());
    state.detailChartInitialized = true;
  } else {
    Plotly.react(el.detailChart, [trace], layout, plotlyConfig());
  }
}

function renderTrades(trades) {
  if (!trades.length) {
    el.tradesTableBody.innerHTML = `<tr><td colspan="14" class="empty-row">No trades yet.</td></tr>`;
    return;
  }
  const rows = trades.map((t) => {
    const dir = dirLabel(t.direction);
    return `
      <tr>
        <td>${fmtTime(t.timestamp)}</td>
        <td>${escapeHtml(t.event || "")}</td>
        <td class="${dir.cls}">${dir.text}</td>
        <td>${fmtNum(t.price)}</td>
        <td>${t.qty === null || t.qty === undefined ? "--" : fmtNum(t.qty, 4)}</td>
        <td>${fmtNum(t.sl)}</td>
        <td>${fmtNum(t.tp1)}</td>
        <td>${fmtNum(t.tp2)}</td>
        <td>${fmtNum(t.tp3)}</td>
        <td>${fmtNum(t.composite)}</td>
        <td>${fmtNum(t.atr)}</td>
        <td class="${pnlClass(t.pnl_net)}">${t.pnl_net === null || t.pnl_net === undefined ? "--" : signedFmt(t.pnl_net)}</td>
        <td>${t.equity === null || t.equity === undefined ? "--" : fmtNum(t.equity)}</td>
        <td>${escapeHtml(t.note || "")}</td>
      </tr>
    `;
  });
  el.tradesTableBody.innerHTML = rows.join("");
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
