import { useState, useEffect, useRef, useCallback } from "react";
import "./styles/capflex.css";

const CLUSTER_COLORS = [
  "#00FFB2", "#FF4D6D", "#4D9FFF", "#FFB800", "#BF5FFF",
  "#FF7A3D", "#00E0FF", "#FF3DE8", "#7FFF00", "#FF9AAA",
];

const BG = "#080C14";
const BORDER = "#1A2640";
const ACCENT = "#00FFB2";
const MUTED = "#4A6080";

function generateFakeData(n = 150) {
  const points = [];
  for (let i = 0; i < n; i++) {
    const cluster = Math.floor(i / (n / 3));
    const cx = [-2.5, 1.5, 0.5][cluster];
    const cy = [-1, 2, -2.5][cluster];
    points.push({
      id: i,
      x: cx + (Math.random() - 0.5) * 2.5,
      y: cy + (Math.random() - 0.5) * 2.5,
      trueCluster: cluster,
      cluster: null,
      features: {
        f1: +(Math.random() * 5).toFixed(3),
        f2: +(Math.random() * 3).toFixed(3),
        f3: +(Math.random() * 8).toFixed(3),
        f4: +(Math.random() * 4).toFixed(3),
      },
    });
  }
  return points;
}

function generateFakePareto() {
  const points = [];
  for (let i = 0; i < 8; i++) {
    const sil = 0.25 + Math.random() * 0.55;
    points.push({
      solution_id: i + 1,
      silhouette: sil,
      CSVI: Math.max(0, 0.6 - sil * 0.8 + (Math.random() - 0.5) * 0.1),
      ILVC: Math.floor(Math.random() * 8),
      CLVC: Math.floor(Math.random() * 3),
      AMI: 0.5 + Math.random() * 0.4,
      cardinality: `${48 + Math.floor(Math.random() * 4)}-${49 + Math.floor(Math.random() * 3)}-${48 + Math.floor(Math.random() * 5)}`,
      isKnee: false,
    });
  }
  // Sort by silhouette desc, mark knee
  points.sort((a, b) => b.silhouette - a.silhouette);
  points[Math.floor(points.length / 2)].isKnee = true;
  return points;
}

// Main App
export default function CapFlexUI() {
  const [activeTab, setActiveTab] = useState("pca");
  const [inputType, setInputType] = useState("tabular");
  const [fileName, setFileName] = useState(null);
  const [labelCol, setLabelCol] = useState("Species");
  const [targetCard, setTargetCard] = useState("50,50,50");
  const [delta, setDelta] = useState(0.1);
  const [maxIter, setMaxIter] = useState("");
  const [embPrefix, setEmbPrefix] = useState("emb");

  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [statusMsg, setStatusMsg] = useState("No data loaded");
  const [progress, setProgress] = useState(0);

  const [points, setPoints] = useState([]);
  const [clustered, setClustered] = useState(false);
  const [pareto, setPareto] = useState([]);
  const [kneeMetrics, setKneeMetrics] = useState(null);

  const [clusterFilter, setClusterFilter] = useState(null);
  const [sortCol, setSortCol] = useState(null);
  const [sortDir, setSortDir] = useState("asc");
  const [tooltip, setTooltip] = useState({ visible: false, x: 0, y: 0, data: null });

  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const particlesRef = useRef([]);
  //FAKE FILE FOR TESTING
  useEffect(() => {
  const data = generateFakeData(150);
  setPoints(data);
  setStatus("done");
  setStatusMsg("Fake data loaded");
}, []);
  // Load fake data on file select
  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFileName(f.name);
    setStatus("loading");
    setStatusMsg("Parsing and computing PCA projection…");
    setProgress(0);
    setClustered(false);
    setPareto([]);
    setKneeMetrics(null);

    let p = 0;
    const iv = setInterval(() => {
      p += 15 + Math.random() * 20;
      setProgress(Math.min(p, 95));
      if (p >= 95) clearInterval(iv);
    }, 120);

    setTimeout(() => {
      clearInterval(iv);
      setProgress(100);
      const data = generateFakeData(150);
      setPoints(data);
      setStatus("done");
      setStatusMsg(`${data.length} instances loaded — PCA projection ready`);
    }, 900);
  };

  // Run clustering
  const handleRun = () => {
    if (!points.length) return;
    setStatus("loading");
    setStatusMsg("Dispatching clustering job to capflex-svc…");
    setProgress(0);
    setClustered(false);
    setPareto([]);
    setKneeMetrics(null);

    const stages = [
      [300, "Generating cardinality pool…", 20],
      [700, "Running parallel LP optimization…", 55],
      [900, "Computing Pareto front…", 80],
      [600, "Identifying knee point…", 92],
      [400, "Reconstructing final model…", 100],
    ];

    let delay = 0;
    stages.forEach(([dur, msg, prog]) => {
      delay += dur;
      setTimeout(() => {
        setStatusMsg(msg);
        setProgress(prog);
      }, delay);
    });

    setTimeout(() => {
      const k = targetCard.split(",").length;
      const newPoints = points.map((pt) => ({
        ...pt,
        cluster: Math.floor(Math.random() * k),
      }));
      setPoints(newPoints);
      setClustered(true);
      setPareto(generateFakePareto());
      setKneeMetrics({
        silhouette: 0.672,
        CSVI: 0.041,
        ILVC: 3,
        CLVC: 1,
        AMI: 0.814,
      });
      setStatus("done");
      setStatusMsg("Optimal solution identified — Knee point selected");
    }, delay + 200);
  };

  //Canvas drawing 
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const PAD = 48;

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);

    if (!points.length) return;

    // Compute bounds
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const xRange = xmax - xmin || 1;
    const yRange = ymax - ymin || 1;

    const toCanvas = (px, py) => ({
      cx: PAD + ((px - xmin) / xRange) * (W - PAD * 2),
      cy: H - PAD - ((py - ymin) / yRange) * (H - PAD * 2),
    });

    // Grid lines
    ctx.strokeStyle = BORDER + "88";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const x = PAD + (i / 5) * (W - PAD * 2);
      const y = PAD + (i / 5) * (H - PAD * 2);
      ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, H - PAD); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = BORDER;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(PAD, PAD); ctx.lineTo(PAD, H - PAD); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD, H - PAD); ctx.lineTo(W - PAD, H - PAD); ctx.stroke();

    // Draw points
    points.forEach((pt) => {
      const { cx, cy } = toCanvas(pt.x, pt.y);
      const isFiltered = clusterFilter !== null && pt.cluster !== clusterFilter;
      let color = "#3A4A60";
      if (clustered && pt.cluster !== null) {
        color = CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length];
      }

      ctx.globalAlpha = isFiltered ? 0.12 : 0.9;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      if (!isFiltered) {
        ctx.strokeStyle = clustered ? color + "44" : "#1A2640";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
    ctx.globalAlpha = 1;
  }, [points, clustered, clusterFilter]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      drawCanvas();
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [drawCanvas]);

  useEffect(() => { drawCanvas(); }, [drawCanvas]);

  //Canvas hover for tooltip
  const handleCanvasMouseMove = (e) => {
    if (!points.length || !clustered) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const W = canvas.width, H = canvas.height, PAD = 48;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);

    let nearest = null, minDist = 14;
    points.forEach((pt) => {
      const cx = PAD + ((pt.x - xmin) / (xmax - xmin || 1)) * (W - PAD * 2);
      const cy = H - PAD - ((pt.y - ymin) / (ymax - ymin || 1)) * (H - PAD * 2);
      const d = Math.hypot(mx - cx, my - cy);
      if (d < minDist) { minDist = d; nearest = pt; }
    });

    if (nearest) {
      setTooltip({ visible: true, x: e.clientX + 12, y: e.clientY - 10, data: nearest });
    } else {
      setTooltip((t) => ({ ...t, visible: false }));
    }
  };

  //Table logic
  const tableData = clustered
    ? points
        .filter((p) => clusterFilter === null || p.cluster === clusterFilter)
        .sort((a, b) => {
          if (!sortCol) return 0;
          const va = sortCol === "cluster" ? a.cluster : a.features?.[sortCol] ?? 0;
          const vb = sortCol === "cluster" ? b.cluster : b.features?.[sortCol] ?? 0;
          return sortDir === "asc" ? va - vb : vb - va;
        })
    : [];

  const handleSort = (col) => {
    if (sortCol === col) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortCol(col); setSortDir("asc"); }
  };

  const clusters = clustered
    ? [...new Set(points.map((p) => p.cluster))].sort((a, b) => a - b)
    : [];

  const knee = pareto.find((p) => p.isKnee);

  return (
    <>

      <div className="app">
        {/* Header */}
        <header className="header">
          <div className="logo">Cap<span>Flex</span></div>
          <div className="badge">2026</div>
          <div style={{ fontSize: 12, color: MUTED, fontFamily: "'Space Mono', monospace" }}>
            Semi-supervised Flexible Cardinality Clustering
          </div>
          <nav className="tabs">
            <button className={`tab ${activeTab === "pca" ? "active" : ""}`} onClick={() => setActiveTab("pca")}>
              PCA Explorer
            </button>
            <button
              className={`tab ${activeTab === "table" ? "active" : ""}`}
              onClick={() => setActiveTab("table")}
              disabled={!clustered}
              style={{ opacity: clustered ? 1 : 0.3 }}
            >
              Cluster Results
            </button>
          </nav>
        </header>

        <div className="main">
          {/* Sidebar */}
          <aside className="sidebar">
            {/* Upload */}
            <div className="section">
              <div className="section-title">Data Source</div>
              <div className="input-type-group">
                {["tabular", "embeddings"].map((t) => (
                  <button
                    key={t}
                    className={`input-type-btn ${inputType === t ? "active" : ""}`}
                    onClick={() => setInputType(t)}
                  >
                    {t === "tabular" ? "TABULAR" : "EMBEDDINGS"}
                  </button>
                ))}
              </div>
              <div className={`upload-zone ${fileName ? "has-file" : ""}`}>
                <input type="file" accept=".csv" onChange={handleFileChange} />
                <div className="upload-icon">{fileName ? "📄" : "⬆"}</div>
                <div className="upload-text">
                  {fileName ? "File loaded" : "Drop CSV here or click to browse"}
                </div>
                {fileName && <div className="upload-filename">{fileName}</div>}
              </div>
            </div>

            {/* Parameters */}
            <div className="section">
              <div className="section-title">Parameters</div>

              {inputType === "tabular" && (
                <div className="form-group">
                  <div className="form-label">
                    LABEL COLUMN <span className="hint">optional</span>
                  </div>
                  <input
                    type="text"
                    value={labelCol}
                    onChange={(e) => setLabelCol(e.target.value)}
                    placeholder="e.g. Species"
                  />
                </div>
              )}

              {inputType === "embeddings" && (
                <div className="form-group">
                  <div className="form-label">EMBEDDING PREFIX</div>
                  <input
                    type="text"
                    value={embPrefix}
                    onChange={(e) => setEmbPrefix(e.target.value)}
                    placeholder="emb"
                  />
                </div>
              )}

              <div className="form-group">
                <div className="form-label">
                  TARGET CARDINALITY
                  <span className="hint">comma-separated</span>
                </div>
                <input
                  type="text"
                  value={targetCard}
                  onChange={(e) => setTargetCard(e.target.value)}
                  placeholder="50,50,50"
                />
              </div>

              <div className="form-group">
                <div className="form-label">
                  DELTA (δ) <span className="hint">tolerance</span>
                </div>
                <div className="range-row">
                  <input
                    type="range"
                    min="0" max="0.5" step="0.01"
                    value={delta}
                    onChange={(e) => setDelta(+e.target.value)}
                  />
                  <span className="range-val">{delta.toFixed(2)}</span>
                </div>
              </div>

              <div className="form-group">
                <div className="form-label">
                  MAX ITERATIONS <span className="hint">auto if empty</span>
                </div>
                <input
                  type="number"
                  value={maxIter}
                  onChange={(e) => setMaxIter(e.target.value)}
                  placeholder="auto"
                />
              </div>
            </div>

            {/* Run */}
            <div className="section">
              <button
                className="run-btn"
                onClick={handleRun}
                disabled={!points.length || status === "loading"}
              >
                {status === "loading" ? (
                  <><span className="shimmer" />Running…</>
                ) : (
                  "▶ Run CapFlex"
                )}
              </button>
            </div>

            {/* Knee point summary */}
            {kneeMetrics && (
              <div className="section">
                <div className="section-title">Knee Point Solution</div>
                {[
                  ["Silhouette", kneeMetrics.silhouette.toFixed(4)],
                  ["CSVI", kneeMetrics.CSVI.toFixed(4)],
                  ["ILVC", kneeMetrics.ILVC],
                  ["CLVC", kneeMetrics.CLVC],
                  ["AMI", kneeMetrics.AMI.toFixed(4)],
                ].map(([k, v]) => (
                  <div key={k} className="pareto-metric" style={{ marginBottom: 6 }}>
                    <span className="k" style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, color: MUTED }}>{k}</span>
                    <span className="v" style={{ fontFamily: "'Space Mono',monospace", fontSize: 12, color: ACCENT }}>{v}</span>
                  </div>
                ))}
              </div>
            )}
          </aside>

          {/* Content area */}
          <div className="content">
            {/* Status bar */}
            <div className="status-bar">
              <div className={`status-dot ${status === "loading" ? "active" : status === "done" ? "done" : status === "error" ? "error" : ""}`} />
              <span>{statusMsg}</span>
              {status === "loading" && (
                <span style={{ marginLeft: "auto", color: ACCENT, fontFamily: "'Space Mono',monospace", fontSize: 11 }}>
                  {Math.round(progress)}%
                </span>
              )}
            </div>
            {/* Progress bar */}
            <div className="progress-bar">
              {status === "loading" ? (
                progress < 20 ? (
                  <div className="progress-fill indeterminate" />
                ) : (
                  <div className="progress-fill" style={{ width: `${progress}%` }} />
                )
              ) : (
                <div className="progress-fill" style={{ width: status === "done" ? "100%" : "0%", opacity: 0.3 }} />
              )}
            </div>

            {/* PCA Tab */}
            {activeTab === "pca" && (
              <>
                <div className="pca-area">
                  <canvas
                    ref={canvasRef}
                    className="pca-canvas"
                    onMouseMove={handleCanvasMouseMove}
                    onMouseLeave={() => setTooltip((t) => ({ ...t, visible: false }))}
                  />

                  {/* Axis labels */}
                  {points.length > 0 && (
                    <>
                      <div className="axis-label" style={{ bottom: 8, left: "50%", transform: "translateX(-50%)" }}>
                        PC1 ({(Math.random() * 10 + 35).toFixed(1)}% variance)
                      </div>
                      <div className="axis-label" style={{ left: 4, top: "50%", transform: "rotate(-90deg) translateX(-50%)", transformOrigin: "left center" }}>
                        PC2 ({(Math.random() * 5 + 18).toFixed(1)}%)
                      </div>
                    </>
                  )}

                  {/* Empty overlay */}
                  {!points.length && (
                    <div className="pca-overlay">
                      <div className="pca-placeholder">
                        <div className="big">◎</div>
                        <p>Load a CSV to visualize PCA projection</p>
                      </div>
                    </div>
                  )}

                  {/* Legend */}
                  {clustered && clusters.length > 0 && (
                    <div className="legend">
                      <div className="legend-title">Clusters</div>
                      {clusters.map((c) => (
                        <div
                          key={c}
                          className="legend-item"
                          style={{ cursor: "pointer", opacity: clusterFilter === null || clusterFilter === c ? 1 : 0.35 }}
                          onClick={() => setClusterFilter(clusterFilter === c ? null : c)}
                        >
                          <div className="legend-dot" style={{ background: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }} />
                          Cluster {c + 1}
                          <span style={{ color: MUTED, fontSize: 10, marginLeft: 4 }}>
                            ({points.filter((p) => p.cluster === c).length})
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Pre-cluster state */}
                  {points.length > 0 && !clustered && (
                    <div className="legend">
                      <div className="legend-title">State</div>
                      <div className="legend-item">
                        <div className="legend-dot" style={{ background: "#3A4A60" }} />
                        Unclustered ({points.length})
                      </div>
                    </div>
                  )}
                </div>

                {/* Pareto panel */}
                {pareto.length > 0 && (
                  <div className="pareto-panel">
                    <div className="pareto-header">
                      <div className="pareto-title">Pareto Front — Non-Dominated Solutions</div>
                      {knee && (
                        <div className="pareto-knee">
                          Knee: Sol. {knee.solution_id} · Sil {knee.silhouette.toFixed(3)} · CSVI {knee.CSVI.toFixed(3)}
                        </div>
                      )}
                    </div>
                    <div className="pareto-body">
                      {pareto.map((sol) => (
                        <div key={sol.solution_id} className={`pareto-card ${sol.isKnee ? "knee" : ""}`}>
                          <div className="pareto-card-id">
                            {sol.isKnee ? "★ KNEE" : `SOL. ${sol.solution_id}`}
                          </div>
                          <div className="pareto-metric">
                            <span className="k">Sil</span>
                            <span className="v good">{sol.silhouette.toFixed(3)}</span>
                          </div>
                          <div className="pareto-metric">
                            <span className="k">CSVI</span>
                            <span className="v">{sol.CSVI.toFixed(3)}</span>
                          </div>
                          <div className="pareto-metric">
                            <span className="k">AMI</span>
                            <span className="v">{sol.AMI.toFixed(3)}</span>
                          </div>
                          <div className="pareto-metric">
                            <span className="k">ILVC / CLVC</span>
                            <span className="v">{sol.ILVC} / {sol.CLVC}</span>
                          </div>
                          <div className="cardinality">Card: {sol.cardinality}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Table Tab */}
            {activeTab === "table" && (
              <>
                {clustered && kneeMetrics && (
                  <div className="metrics-row">
                    {[
                      ["Silhouette", kneeMetrics.silhouette.toFixed(4), true],
                      ["CSVI", kneeMetrics.CSVI.toFixed(4), false],
                      ["ILVC", kneeMetrics.ILVC, false],
                      ["CLVC", kneeMetrics.CLVC, false],
                      ["AMI", kneeMetrics.AMI.toFixed(4), true],
                    ].map(([k, v, good]) => (
                      <div key={k} className="metric-cell">
                        <span className="mk">{k}</span>
                        <span className={`mv ${good ? "" : "neutral"}`}>{v}</span>
                      </div>
                    ))}
                  </div>
                )}

                {clustered ? (
                  <>
                    <div className="filter-bar">
                      <span style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, color: MUTED, letterSpacing: 1, flexShrink: 0 }}>FILTER:</span>
                      <div
                        className={`filter-chip ${clusterFilter === null ? "active" : ""}`}
                        style={clusterFilter === null ? { background: "#3A4A60", borderColor: "#3A4A60"} : {}}
                        onClick={() => setClusterFilter(null)}
                      >
                        All ({points.length})
                      </div>
                      {clusters.map((c) => (
                        <div
                          key={c}
                          className={`filter-chip ${clusterFilter === c ? "active" : ""}`}
                          style={clusterFilter === c ? {
                            background: CLUSTER_COLORS[c % CLUSTER_COLORS.length],
                            borderColor: CLUSTER_COLORS[c % CLUSTER_COLORS.length],
                          } : {}}
                          onClick={() => setClusterFilter(clusterFilter === c ? null : c)}
                        >
                          <div className="dot" style={{ background: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }} />
                          Cluster {c + 1} ({points.filter((p) => p.cluster === c).length})
                        </div>
                      ))}
                    </div>

                    <div className="table-area">
                      <table>
                        <thead>
                          <tr>
                            <th onClick={() => handleSort("id")} className={sortCol === "id" ? "sorted" : ""}>
                              ID <span className="sort-icon">{sortCol === "id" ? (sortDir === "asc" ? "↑" : "↓") : "↕"}</span>
                            </th>
                            {Object.keys(points[0]?.features || {}).map((f) => (
                              <th key={f} onClick={() => handleSort(f)} className={sortCol === f ? "sorted" : ""}>
                                {f.toUpperCase()} <span className="sort-icon">{sortCol === f ? (sortDir === "asc" ? "↑" : "↓") : "↕"}</span>
                              </th>
                            ))}
                            <th onClick={() => handleSort("cluster")} className={sortCol === "cluster" ? "sorted" : ""}>
                              CLUSTER <span className="sort-icon">{sortCol === "cluster" ? (sortDir === "asc" ? "↑" : "↓") : "↕"}</span>
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {tableData.slice(0, 300).map((pt) => (
                            <tr key={pt.id}>
                              <td>{pt.id}</td>
                              {Object.values(pt.features).map((v, i) => (
                                <td key={i}>{v}</td>
                              ))}
                              <td>
                                <span
                                  className="cluster-badge"
                                  style={{
                                    background: CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length] + "20",
                                    color: CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length],
                                  }}
                                >
                                  <span className="dot" style={{ background: CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length] }} />
                                  {pt.cluster + 1}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {tableData.length > 300 && (
                        <div style={{ padding: "12px 0", textAlign: "center", fontFamily: "'Space Mono',monospace", fontSize: 11, color: MUTED }}>
                          Showing 300 of {tableData.length} rows
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="empty-state">
                    <div className="big">⬡</div>
                    <div>Run clustering first to see assignments</div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Tooltip */}
      <div
        className={`tooltip ${tooltip.visible ? "visible" : ""}`}
        style={{ left: tooltip.x, top: tooltip.y }}
      >
        {tooltip.data && (
          <>
            <div className="tcluster">Cluster {(tooltip.data.cluster ?? "?") + 1}</div>
            <div>ID: {tooltip.data.id}</div>
            {Object.entries(tooltip.data.features || {}).map(([k, v]) => (
              <div key={k}>{k}: {v}</div>
            ))}
          </>
        )}
      </div>
    </>
  );
}