/**
 * CropNow - AI Crop Failure & Yield-Loss Early Warning System
 * Frontend Application Engine
 */

// Global State
let currentDatasetId = "demo-84-fields";
let currentPredictionId = "demo-84-fields";
let allPredictions = [];
let portfolioSummary = null;
let currentFieldDetail = null;
let currentSortCol = "priority_rank";
let currentSortAsc = true;
let uploadMappingSuggestion = {};
let uploadedCSVColumns = [];
let rawDatasetRecords = [];
let rawDatasetStats = {};

// Chart Instances
let chartRiskDist = null;
let chartCropComp = null;
let chartFieldTraj = null;
let chartSimComp = null;
let chartFeatureImp = null;

// Initialize App
document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  setupDropzone();
  await loadInitialPredictions();
  await loadMLOpsMetrics();
  await loadRawDatasetData();
});

// Switch Tabs
function switchTab(tabId) {
  document.querySelectorAll(".nav-tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-view").forEach(view => view.classList.remove("active"));

  const targetTab = document.getElementById(`tab-${tabId}`);
  if (targetTab) {
    targetTab.classList.add("active");
  }

  // Highlight button
  const matchingBtn = Array.from(document.querySelectorAll(".nav-tab-btn")).find(btn => 
    btn.getAttribute("onclick") && btn.getAttribute("onclick").includes(tabId)
  );
  if (matchingBtn) matchingBtn.classList.add("active");

  if (tabId === "sandbox") {
    setTimeout(runLiveSimulation, 50);
  }
  if (tabId === "raw-data") {
    loadRawDatasetData();
  }

  if (window.lucide) {
    lucide.createIcons();
  }
}

// Show Toast Notification
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = "toast";
  
  let icon = "info";
  if (type === "success") icon = "check-circle";
  if (type === "error") icon = "alert-circle";
  if (type === "warning") icon = "alert-triangle";

  toast.innerHTML = `
    <i data-lucide="${icon}" style="color: ${type === 'error' ? 'var(--coral-light)' : 'var(--emerald-light)'}"></i>
    <span>${message}</span>
  `;
  container.appendChild(toast);
  if (window.lucide) lucide.createIcons();

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Load Initial Predictions from Backend
async function loadInitialPredictions() {
  try {
    const res = await fetch(`/api/predictions/${currentPredictionId}`);
    if (!res.ok) throw new Error("Failed to fetch initial prediction dataset.");
    const data = await res.json();
    
    allPredictions = data.results || [];
    portfolioSummary = data.portfolio_summary;
    
    renderPortfolioView();
    renderRiskTable();
    populateFieldSelectors();
    
    if (allPredictions.length > 0) {
      loadFieldDeepDive(allPredictions[0].field_id);
    }
  } catch (err) {
    console.error("Init error:", err);
    showToast("Connecting to CropNow ML Engine...", "info");
  }
}

// Format Indian Rupee (INR) with Indian numbering (Lakhs, Crores or commas)
function formatINR(val) {
  if (typeof val !== 'number') val = parseFloat(val) || 0;
  if (val >= 10000000) {
    return `₹${(val / 10000000).toFixed(2)} Cr`;
  } else if (val >= 100000) {
    return `₹${(val / 100000).toFixed(2)} Lakhs`;
  } else {
    return `₹${Math.round(val).toLocaleString('en-IN')}`;
  }
}

// Render Portfolio View
function renderPortfolioView() {
  updatePortfolioMetrics(portfolioSummary);
}

// Update Portfolio Overview Metrics & KPI Cards
function updatePortfolioMetrics(portfolioSummary) {
  if (!portfolioSummary) return;

  // Headline alert banner
  const headlineEl = document.getElementById("portfolio-headline-text");
  if (headlineEl) headlineEl.textContent = portfolioSummary.portfolio_headline;

  // KPIs
  document.getElementById("kpi-total-fields").textContent = portfolioSummary.total_fields;
  document.getElementById("kpi-high-critical-count").textContent = portfolioSummary.high_critical_count;
  const critPct = Math.round((portfolioSummary.high_critical_count / portfolioSummary.total_fields) * 100);
  document.getElementById("kpi-high-critical-pct").textContent = `${critPct}% of monitored portfolio`;
  document.getElementById("kpi-avg-loss-pct").textContent = `${portfolioSummary.average_loss_pct}%`;
  
  const lossInr = portfolioSummary.estimated_loss_exposure_inr || portfolioSummary.estimated_loss_exposure_usd || 0;
  document.getElementById("kpi-loss-usd").textContent = formatINR(lossInr);
  document.getElementById("kpi-loss-tons").textContent = `${portfolioSummary.total_loss_exposure_tons.toLocaleString()} t potential yield gap`;

  // Top Priority List Preview
  const tbody = document.getElementById("top-priority-table-body");
  const topList = allPredictions.slice(0, 5);
  
  tbody.innerHTML = topList.map(item => `
    <tr onclick="selectFieldAndDeepDive('${item.field_id}')">
      <td><span class="badge badge-urgent">#${item.priority_rank}</span></td>
      <td style="font-weight: 700; color: var(--emerald-light);">${item.field_id}</td>
      <td><strong>${item.crop_type}</strong> <span style="font-size: 11px; color: var(--text-dim);">${item.variety}</span></td>
      <td><strong>${item.predicted_yield}</strong> <span style="font-size: 11px; color: var(--text-dim);">/ ${item.expected_yield} t/ac</span></td>
      <td style="font-weight: 700; color: ${item.yield_loss_percentage > 25 ? 'var(--coral-light)' : 'var(--amber-light)'};">
        -${item.yield_loss_percentage}%
      </td>
      <td><span class="badge badge-${item.risk_level.toLowerCase()}">${Math.round(item.failure_probability * 100)}% (${item.risk_level})</span></td>
      <td><span style="font-size: 12px;">${item.primary_risk_factor}</span></td>
      <td><span style="color: var(--amber-light); font-weight: 600;">${item.intervention_window}</span></td>
      <td><button class="btn btn-secondary btn-sm" onclick="event.stopPropagation(); selectFieldAndDeepDive('${item.field_id}')">Deep-Dive &rarr;</button></td>
    </tr>
  `).join("");

  renderPortfolioCharts();
}

// Render Portfolio Charts (Risk Doughnut & Crop Bar Chart)
function renderPortfolioCharts() {
  if (!portfolioSummary) return;

  // Chart 1: Risk Distribution Doughnut
  const ctxRisk = document.getElementById("chart-risk-distribution").getContext("2d");
  if (chartRiskDist) chartRiskDist.destroy();

  const rc = portfolioSummary.risk_counts;
  chartRiskDist = new Chart(ctxRisk, {
    type: "doughnut",
    data: {
      labels: ["Critical Risk", "High Risk", "Medium Risk", "Low Risk"],
      datasets: [{
        data: [rc.critical, rc.high, rc.medium, rc.low],
        backgroundColor: [
          "#ef4444", // Coral Red
          "#f97316", // Orange
          "#f59e0b", // Amber
          "#10b981"  // Emerald
        ],
        borderWidth: 2,
        borderColor: "#0e1711"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "right",
          labels: { color: "#94a3b8", font: { family: "Inter", size: 12 } }
        }
      },
      cutout: "68%"
    }
  });

  // Chart 2: Expected vs Predicted Yield by Crop
  const ctxCrop = document.getElementById("chart-crop-comparison").getContext("2d");
  if (chartCropComp) chartCropComp.destroy();

  const crops = ["Corn", "Soybean", "Wheat", "Cotton", "Rice"];
  const expectedAverages = [];
  const predictedAverages = [];

  crops.forEach(crop => {
    const cropRecords = allPredictions.filter(p => p.crop_type === crop);
    if (cropRecords.length > 0) {
      const expAvg = cropRecords.reduce((acc, curr) => acc + curr.expected_yield, 0) / cropRecords.length;
      const predAvg = cropRecords.reduce((acc, curr) => acc + curr.predicted_yield, 0) / cropRecords.length;
      expectedAverages.push(Number(expAvg.toFixed(2)));
      predictedAverages.push(Number(predAvg.toFixed(2)));
    } else {
      expectedAverages.push(0);
      predictedAverages.push(0);
    }
  });

  chartCropComp = new Chart(ctxCrop, {
    type: "bar",
    data: {
      labels: crops,
      datasets: [
        {
          label: "Expected Potential (t/ac)",
          data: expectedAverages,
          backgroundColor: "rgba(52, 211, 153, 0.4)",
          borderColor: "#34d399",
          borderWidth: 1.5,
          borderRadius: 4
        },
        {
          label: "Model Predicted (t/ac)",
          data: predictedAverages,
          backgroundColor: "rgba(245, 158, 11, 0.7)",
          borderColor: "#f59e0b",
          borderWidth: 1.5,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(52, 211, 153, 0.08)" } }
      },
      plugins: {
        legend: { labels: { color: "#94a3b8" } }
      }
    }
  });
}

// Populate Selectors
function populateFieldSelectors() {
  const deepDiveSelect = document.getElementById("deepdive-field-selector");
  const simSelect = document.getElementById("sim-field-selector");
  
  const optionsHtml = allPredictions.map(item => `
    <option value="${item.field_id}">${item.field_id} — ${item.crop_type} (${item.risk_level})</option>
  `).join("");

  if (deepDiveSelect) deepDiveSelect.innerHTML = optionsHtml;
  if (simSelect) simSelect.innerHTML = optionsHtml;
}

// Render Prioritized Risk Matrix Table
function renderRiskTable() {
  const tbody = document.getElementById("full-risk-table-body");
  const search = (document.getElementById("table-search")?.value || "").toLowerCase();
  const riskFilter = document.getElementById("filter-risk")?.value || "ALL";
  const cropFilter = document.getElementById("filter-crop")?.value || "ALL";

  let filtered = allPredictions.filter(item => {
    const matchSearch = !search || 
      item.field_id.toLowerCase().includes(search) || 
      item.crop_type.toLowerCase().includes(search) || 
      item.variety.toLowerCase().includes(search) || 
      item.primary_risk_factor.toLowerCase().includes(search);
      
    const matchRisk = riskFilter === "ALL" || item.risk_level.toUpperCase() === riskFilter;
    const matchCrop = cropFilter === "ALL" || item.crop_type.toLowerCase() === cropFilter.toLowerCase();
    return matchSearch && matchRisk && matchCrop;
  });

  // Sort
  filtered.sort((a, b) => {
    let valA = a[currentSortCol];
    let valB = b[currentSortCol];
    if (typeof valA === "string") {
      return currentSortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return currentSortAsc ? valA - valB : valB - valA;
  });

  document.getElementById("table-record-count").textContent = `Showing ${filtered.length} of ${allPredictions.length} fields`;

  tbody.innerHTML = filtered.map(item => `
    <tr onclick="selectFieldAndDeepDive('${item.field_id}')">
      <td><span class="badge ${item.priority_rank <= 10 ? 'badge-urgent' : 'badge-low'}">#${item.priority_rank}</span></td>
      <td style="font-weight: 700; color: var(--emerald-light);">${item.field_id}</td>
      <td><strong>${item.crop_type}</strong> <div style="font-size: 11px; color: var(--text-dim);">${item.variety}</div></td>
      <td><span style="font-size: 12px; color: var(--text-muted);">${item.growth_stage} (${item.crop_age_days}d)</span></td>
      <td>${item.expected_yield} t/ac</td>
      <td style="font-weight: 700;">${item.predicted_yield} t/ac</td>
      <td style="font-weight: 700; color: ${item.yield_loss_percentage > 25 ? 'var(--coral-light)' : 'var(--amber-light)'};">
        -${item.yield_loss_percentage}%
      </td>
      <td>${Math.round(item.failure_probability * 100)}%</td>
      <td><span class="badge badge-${item.risk_level.toLowerCase()}">${item.risk_level}</span></td>
      <td><span style="font-size: 12px; font-weight: 600;">${item.primary_risk_factor}</span></td>
      <td><span style="color: var(--amber-light); font-weight: 600;">${item.intervention_window}</span></td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="event.stopPropagation(); selectFieldAndDeepDive('${item.field_id}')">
          Deep-Dive
        </button>
      </td>
    </tr>
  `).join("");
}

function filterRiskTable() {
  renderRiskTable();
}

function sortTable(col) {
  if (currentSortCol === col) {
    currentSortAsc = !currentSortAsc;
  } else {
    currentSortCol = col;
    currentSortAsc = true;
  }
  renderRiskTable();
}

// Select Field & Jump to Deep-Dive Tab
function selectFieldAndDeepDive(fieldId) {
  const select = document.getElementById("deepdive-field-selector");
  if (select) select.value = fieldId;
  loadFieldDeepDive(fieldId);
  switchTab("field-deepdive");
}

// Load Field Deep-Dive with SHAP & 7-Day Trajectory
async function loadFieldDeepDive(fieldId) {
  try {
    const res = await fetch(`/api/field/${currentPredictionId}/${fieldId}`);
    if (!res.ok) throw new Error("Could not load field details.");
    const field = await res.json();
    currentFieldDetail = field;

    // Header info
    document.getElementById("dive-field-name").textContent = `Field ${field.field_id}`;
    document.getElementById("dive-crop-variety").textContent = `${field.crop_type} (${field.variety}) • ${field.growth_stage} (${field.crop_age_days} days)`;
    document.getElementById("dive-yield-stats").textContent = `${field.expected_yield} → ${field.predicted_yield} t/ac`;
    document.getElementById("dive-loss-pct").textContent = `-${field.yield_loss_percentage}% Estimated Yield Loss`;
    document.getElementById("dive-fail-prob").textContent = `${Math.round(field.failure_probability * 100)}%`;
    
    const riskBadge = document.getElementById("dive-risk-badge");
    riskBadge.className = `badge badge-${field.risk_level.toLowerCase()}`;
    riskBadge.textContent = field.risk_level;

    document.getElementById("dive-window").textContent = field.intervention_window;

    // Render SHAP Factor Breakdown Bars
    renderSHAPBars(field.shap_breakdown);

    // Render 7-Day Trajectory Chart
    renderTrajectoryChart(field.risk_trajectory);

    // Render Recommendations
    const recsList = document.getElementById("field-recommendations-list");
    recsList.innerHTML = field.recommended_actions.map(action => `
      <div style="display: flex; align-items: flex-start; gap: 10px; background: var(--bg-input); padding: 12px; border-radius: var(--radius-md); border-left: 3px solid var(--emerald-primary);">
        <i data-lucide="check-circle-2" style="color: var(--emerald-light); flex-shrink: 0; width: 18px; height: 18px; margin-top: 2px;"></i>
        <div style="font-size: 13px; color: var(--text-main); font-weight: 500;">${action}</div>
      </div>
    `).join("");

    // Render Sensor Telemetry Grid
    const sensorGrid = document.getElementById("field-sensor-grid");
    const s = field.sensors;
    sensorGrid.innerHTML = `
      <div style="background: var(--bg-input); padding: 10px 14px; border-radius: var(--radius-md);">
        <span style="color: var(--text-dim); font-size: 11px;">SOIL MOISTURE</span>
        <div style="font-weight: 700; color: ${s.soil_moisture < 30 ? 'var(--coral-light)' : 'var(--emerald-light)'}">${s.soil_moisture}%</div>
      </div>
      <div style="background: var(--bg-input); padding: 10px 14px; border-radius: var(--radius-md);">
        <span style="color: var(--text-dim); font-size: 11px;">SOIL TEMP / pH</span>
        <div style="font-weight: 700;">${s.soil_temperature}&deg;C / ${s.soil_ph}</div>
      </div>
      <div style="background: var(--bg-input); padding: 10px 14px; border-radius: var(--radius-md);">
        <span style="color: var(--text-dim); font-size: 11px;">AIR TEMP / HUMIDITY</span>
        <div style="font-weight: 700;">${s.temperature}&deg;C / ${s.humidity}%</div>
      </div>
      <div style="background: var(--bg-input); padding: 10px 14px; border-radius: var(--radius-md);">
        <span style="color: var(--text-dim); font-size: 11px;">CANOPY NDVI</span>
        <div style="font-weight: 700; color: ${s.ndvi < 0.6 ? 'var(--coral-light)' : 'var(--emerald-light)'}">${s.ndvi}</div>
      </div>
      <div style="background: var(--bg-input); padding: 10px 14px; border-radius: var(--radius-md);">
        <span style="color: var(--text-dim); font-size: 11px;">DISEASE SEVERITY</span>
        <div style="font-weight: 700; color: ${s.disease_score > 25 ? 'var(--coral-light)' : 'var(--text-main)'}">${s.disease_score}/100</div>
      </div>
      <div style="background: var(--bg-input); padding: 10px 14px; border-radius: var(--radius-md);">
        <span style="color: var(--text-dim); font-size: 11px;">N - P - K LEVELS</span>
        <div style="font-weight: 700;">${s.nitrogen} - ${s.phosphorus} - ${s.potassium} kg/ha</div>
      </div>
    `;

    if (window.lucide) lucide.createIcons();
  } catch (err) {
    console.error("Deep-dive error:", err);
  }
}

// Render SHAP Attribution Percentage Bars
function renderSHAPBars(shap) {
  const container = document.getElementById("shap-breakdown-bars");
  const factors = [
    { name: "Disease Pressure", pct: shap.disease_pressure, color: "#ef4444" },
    { name: "Water Stress", pct: shap.water_stress, color: "#34d399" },
    { name: "Weather & Heat", pct: shap.weather, color: "#f59e0b" },
    { name: "Nutrient Imbalance", pct: shap.nutrition, color: "#84cc16" },
    { name: "Pest Pressure", pct: shap.pest_pressure, color: "#f97316" },
    { name: "Other / Baseline", pct: shap.other, color: "#64748b" }
  ];

  factors.sort((a, b) => b.pct - a.pct);

  container.innerHTML = factors.map(f => `
    <div class="shap-bar-item">
      <div class="shap-label-row">
        <span>${f.name}</span>
        <span style="color: ${f.color}; font-weight: 700;">${f.pct}%</span>
      </div>
      <div class="shap-track">
        <div class="shap-fill" style="width: ${f.pct}%; background-color: ${f.color};"></div>
      </div>
    </div>
  `).join("");
}

// Render 7-Day Forward Trajectory Chart
function renderTrajectoryChart(trajectory) {
  const ctxTraj = document.getElementById("chart-field-trajectory").getContext("2d");
  if (chartFieldTraj) chartFieldTraj.destroy();

  const labels = trajectory.map(t => t.day);
  const data = trajectory.map(t => t.risk_pct);

  chartFieldTraj = new Chart(ctxTraj, {
    type: "line",
    data: {
      labels: labels,
      datasets: [{
        label: "Projected Failure Risk Probability (%)",
        data: data,
        borderColor: "#f87171",
        backgroundColor: "rgba(239, 68, 68, 0.15)",
        fill: true,
        tension: 0.35,
        pointBackgroundColor: "#ef4444",
        pointBorderColor: "#fff",
        pointRadius: 5,
        borderWidth: 2.5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
        y: { 
          min: 0, 
          max: 100, 
          ticks: { color: "#94a3b8", callback: v => `${v}%` },
          grid: { color: "rgba(52, 211, 153, 0.08)" } 
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  // Step Indicators
  const stepsContainer = document.getElementById("trajectory-steps-container");
  stepsContainer.innerHTML = trajectory.map(t => `
    <div class="trajectory-step">
      <div class="step-circle ${t.risk_pct >= 50 ? 'high-risk' : ''}">${t.risk_pct}%</div>
      <span style="font-size: 11px; color: var(--text-dim);">${t.day}</span>
    </div>
  `).join("");
}

// Open Simulation for Current Field
function openSimulationForCurrentField() {
  if (currentFieldDetail) {
    const simSelect = document.getElementById("sim-field-selector");
    if (simSelect) simSelect.value = currentFieldDetail.field_id;
    syncSimulationField();
  }
  switchTab("sandbox");
}

// Sync Simulation Target Field
function syncSimulationField() {
  resetSimulationControls();
  runLiveSimulation();
}

function resetSimulationControls() {
  document.getElementById("sim-irrigation").value = 0;
  document.getElementById("val-sim-irrigation").textContent = "+0 mm";
  document.getElementById("sim-fert").value = 0;
  document.getElementById("val-sim-fert").textContent = "+0 kg/ha";
  document.getElementById("sim-fungicide").checked = false;
  document.getElementById("val-sim-fungicide").textContent = "Inactive";
  document.getElementById("sim-temp").value = 0;
  document.getElementById("val-sim-temp").textContent = "0 °C";
}

// Simulation Presets Handler
function applySimulationPreset(type) {
  if (type === "rescue") {
    document.getElementById("sim-irrigation").value = 35;
    document.getElementById("sim-fert").value = 50;
    document.getElementById("sim-fungicide").checked = true;
    document.getElementById("sim-temp").value = 0;
  } else if (type === "drought") {
    document.getElementById("sim-irrigation").value = 45;
    document.getElementById("sim-fert").value = 10;
    document.getElementById("sim-fungicide").checked = false;
    document.getElementById("sim-temp").value = -2;
  } else if (type === "disease") {
    document.getElementById("sim-irrigation").value = 10;
    document.getElementById("sim-fert").value = 20;
    document.getElementById("sim-fungicide").checked = true;
    document.getElementById("sim-temp").value = 0;
  }
  runLiveSimulation();
}

// Run Live What-If Counterfactual Simulation
async function runLiveSimulation() {
  const fieldSelect = document.getElementById("sim-field-selector");
  let fieldId = fieldSelect ? fieldSelect.value : "";
  if (!fieldId && allPredictions.length > 0) {
    fieldId = allPredictions[0].field_id;
    if (fieldSelect) fieldSelect.value = fieldId;
  }
  if (!fieldId) return;

  const irrig = parseFloat(document.getElementById("sim-irrigation")?.value || 0);
  const fert = parseFloat(document.getElementById("sim-fert")?.value || 0);
  const fung = document.getElementById("sim-fungicide")?.checked || false;
  const temp = parseFloat(document.getElementById("sim-temp")?.value || 0);

  const valIrrig = document.getElementById("val-sim-irrigation");
  if (valIrrig) valIrrig.textContent = `+${irrig} mm`;
  
  const valFert = document.getElementById("val-sim-fert");
  if (valFert) valFert.textContent = `+${fert} kg/ha`;
  
  const valFung = document.getElementById("val-sim-fungicide");
  if (valFung) valFung.textContent = fung ? "Active (Fungicide Deployed)" : "Inactive";
  
  const valTemp = document.getElementById("val-sim-temp");
  if (valTemp) valTemp.textContent = `${temp > 0 ? '+' : ''}${temp} °C`;

  try {
    const res = await fetch(`/api/simulate?prediction_id=${encodeURIComponent(currentPredictionId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        field_id: fieldId,
        irrigation_adjustment_mm: irrig,
        fertilizer_adjustment_kg: fert,
        fungicide_applied: fung,
        temperature_anomaly: temp
      })
    });

    if (!res.ok) throw new Error("Simulation failed.");
    const sim = await res.json();

    // Metric cards updates
    const elYieldDelta = document.getElementById("sim-yield-delta");
    if (elYieldDelta) elYieldDelta.textContent = `${sim.deltas.yield_gain_tons >= 0 ? '+' : ''}${sim.deltas.yield_gain_tons} t/ac`;
    
    const elYieldDetail = document.getElementById("sim-yield-detail");
    if (elYieldDetail) elYieldDetail.textContent = `${sim.original.predicted_yield} → ${sim.simulated.predicted_yield} t/ac`;

    const elLossDelta = document.getElementById("sim-loss-delta");
    if (elLossDelta) elLossDelta.textContent = `-${sim.deltas.loss_reduction_pct}%`;
    
    const elLossDetail = document.getElementById("sim-loss-detail");
    if (elLossDetail) elLossDetail.textContent = `Remaining: ${sim.simulated.yield_loss_percentage}% loss`;

    const elNewRisk = document.getElementById("sim-new-risk");
    if (elNewRisk) elNewRisk.innerHTML = `<span class="badge badge-${sim.simulated.risk_level.toLowerCase()}">${sim.simulated.risk_level} (${Math.round(sim.simulated.failure_probability * 100)}%)</span>`;
    
    const elRiskDetail = document.getElementById("sim-risk-detail");
    if (elRiskDetail) elRiskDetail.textContent = `Reduced from ${Math.round(sim.original.failure_probability * 100)}% (${sim.original.risk_level})`;

    const elRoiVal = document.getElementById("sim-roi-val");
    if (elRoiVal) elRoiVal.textContent = `+₹${Math.round(sim.deltas.revenue_saved_per_acre).toLocaleString('en-IN')} / ac`;
    
    const elRoiSub = document.getElementById("sim-roi-sub");
    if (elRoiSub) elRoiSub.textContent = `${sim.deltas.roi_multiple}x ROI (Cost: ₹${Math.round(sim.deltas.treatment_cost_per_acre).toLocaleString('en-IN')}/ac)`;

    renderSimulationComparisonChart(sim);
  } catch (err) {
    console.error("Simulation error:", err);
  }
}

// Render Before vs After Comparison Line Chart with Enhanced Aesthetics
function renderSimulationComparisonChart(sim) {
  const ctx = document.getElementById("chart-sim-comparison").getContext("2d");
  if (chartSimComp) chartSimComp.destroy();

  const labels = ["Today (Day 1)", "Day 3 (48h)", "Day 5 (120h)", "Day 7 (168h)"];
  const baselineData = sim.original.risk_trajectory.map(t => t.risk_pct);
  const simulatedData = sim.simulated.risk_trajectory.map(t => t.risk_pct);

  chartSimComp = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Baseline Unmanaged Risk (No Treatment)",
          data: baselineData,
          borderColor: "#ef4444",
          backgroundColor: "rgba(239, 68, 68, 0.05)",
          borderDash: [6, 4],
          tension: 0.35,
          pointRadius: 5,
          pointBackgroundColor: "#ef4444",
          pointBorderColor: "#fff",
          pointBorderWidth: 1.5,
          borderWidth: 2.2
        },
        {
          label: "Mitigated Risk (With Counterfactual Interventions)",
          data: simulatedData,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.18)",
          fill: true,
          tension: 0.35,
          pointRadius: 6,
          pointHoverRadius: 8,
          pointBackgroundColor: "#34d399",
          pointBorderColor: "#0e1711",
          pointBorderWidth: 2,
          borderWidth: 2.8
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      scales: {
        x: { 
          ticks: { color: "#94a3b8", font: { family: "Inter", size: 12, weight: "600" } }, 
          grid: { display: false } 
        },
        y: { 
          min: 0, 
          max: 100, 
          ticks: { 
            color: "#94a3b8", 
            stepSize: 20,
            callback: v => `${v}%` 
          },
          grid: { color: "rgba(52, 211, 153, 0.09)" } 
        }
      },
      plugins: {
        legend: { 
          position: "top",
          labels: { 
            color: "#f8fafc", 
            font: { family: "Inter", size: 12, weight: "600" },
            boxWidth: 14,
            usePointStyle: true
          } 
        },
        tooltip: {
          backgroundColor: "#0e1711",
          titleColor: "#34d399",
          bodyColor: "#f8fafc",
          borderColor: "rgba(52, 211, 153, 0.3)",
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: ${context.raw}% Failure Risk`;
            }
          }
        }
      }
    }
  });
}

// Setup Drag & Drop Upload
function setupDropzone() {
  const dropzone = document.getElementById("csv-dropzone");
  if (!dropzone) return;

  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    }, false);
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    }, false);
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      uploadFiles(files);
    }
  }, false);
}

// Load Quick Scenario from Pre-generated datasets
async function loadQuickScenario(scenario) {
  showToast(`Loading scenario: ${scenario}...`, "info");
  try {
    const res = await fetch(`/api/scenario/${scenario}`);
    if (!res.ok) throw new Error("Could not load scenario.");
    const data = await res.json();
    uploadedDatasetId = data.dataset_id;
    
    // Display Validation Panel
    displayValidationReport(data);
    showToast(`Loaded ${data.filename} (${data.total_rows} rows extracted)`, "success");
  } catch (err) {
    console.error("Scenario load error:", err);
    showToast(`Failed to load scenario: ${err.message}`, "error");
  }
}

function handleFileUpload(event) {
  const files = event.target.files;
  if (files && files.length > 0) uploadFiles(files);
}

// Upload Single or Multi-File Batch to Backend
async function uploadFiles(files) {
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  const label = files.length > 1 ? `${files.length} files (Batch Merge)` : files[0].name;
  showToast(`Uploading & ingesting ${label}...`, "info");

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Document Upload failed.");
    }
    const data = await res.json();

    currentDatasetId = data.dataset_id;
    uploadMappingSuggestion = data.column_mapping_suggestion;
    uploadedCSVColumns = data.columns;

    renderValidationReport(data.validation_report, data.total_rows, data.columns.length);
    showToast(`Successfully parsed and merged ${data.total_rows} observation rows (${data.format})!`, "success");
  } catch (err) {
    console.error("Upload error:", err);
    showToast(err.message, "error");
  }
}

// Render Validation Report Panel
function renderValidationReport(val, rows, cols) {
  const panel = document.getElementById("validation-panel");
  panel.style.display = "block";

  document.getElementById("val-rows").textContent = rows;
  document.getElementById("val-cols").textContent = cols;
  document.getElementById("val-dups").textContent = val.duplicates_count;

  const badge = document.getElementById("validation-status-badge");
  if (val.valid) {
    badge.className = "badge badge-low";
    badge.textContent = "VALID SCHEMA";
  } else {
    badge.className = "badge badge-critical";
    badge.textContent = "ATTENTION REQUIRED";
  }

  const container = document.getElementById("validation-details-container");
  let html = "";

  if (val.missing_columns && val.missing_columns.length > 0) {
    html += `<div style="color: var(--coral-light); margin-bottom: 8px;">
      <i data-lucide="alert-triangle" style="width: 14px; height: 14px; display: inline;"></i> 
      Missing standard columns: <strong>${val.missing_columns.join(", ")}</strong>. Please use Column Mapper.
    </div>`;
  }

  if (val.data_leakage_warnings && val.data_leakage_warnings.length > 0) {
    html += `<div style="color: var(--amber-light); margin-bottom: 8px;">
      <i data-lucide="shield-alert" style="width: 14px; height: 14px; display: inline;"></i> 
      ${val.data_leakage_warnings.join("<br>")}
    </div>`;
  }

  if (val.out_of_range_warnings && val.out_of_range_warnings.length > 0) {
    html += `<div style="color: var(--amber-light); margin-bottom: 8px;">
      <i data-lucide="info" style="width: 14px; height: 14px; display: inline;"></i> 
      Sensor bounds notice: ${val.out_of_range_warnings.length} columns had out-of-physical-range values (automatically clipped).
    </div>`;
  }

  if (!html) {
    html = `<div style="color: var(--emerald-light);"><i data-lucide="check" style="width: 14px; height: 14px; display: inline;"></i> All required agronomic sensor fields detected and ready for ML inference.</div>`;
  }

  container.innerHTML = html;
  if (window.lucide) lucide.createIcons();
}

// Column Mapper Modal
function openColumnMapperModal() {
  const modal = document.getElementById("column-mapper-modal");
  const tbody = document.getElementById("column-mapper-table-body");

  const requiredFields = [
    "field_id", "date", "crop_type", "variety", "crop_age_days", "growth_stage",
    "soil_moisture", "soil_temperature", "soil_ph", "nitrogen", "phosphorus", "potassium",
    "temperature", "humidity", "rainfall", "wind_speed", "solar_radiation",
    "disease_score", "pest_count", "ndvi"
  ];

  tbody.innerHTML = requiredFields.map(field => {
    const suggested = uploadMappingSuggestion[field] || "";
    const options = [`<option value="">-- Unmapped --</option>`]
      .concat(uploadedCSVColumns.map(col => `<option value="${col}" ${col === suggested ? "selected" : ""}>${col}</option>`))
      .join("");

    return `
      <tr>
        <td style="font-weight: 600; color: var(--text-main);">${field}</td>
        <td>
          <select id="map-${field}" class="select-input" style="width: 100%;">
            ${options}
          </select>
        </td>
        <td>
          <span class="badge ${suggested ? 'badge-low' : 'badge-medium'}">
            ${suggested ? 'Matched' : 'Pending'}
          </span>
        </td>
      </tr>
    `;
  }).join("");

  modal.classList.add("active");
  if (window.lucide) lucide.createIcons();
}

function closeColumnMapperModal() {
  document.getElementById("column-mapper-modal").classList.remove("active");
}

async function applyColumnMapping() {
  const mapping = {};
  const selects = document.querySelectorAll("[id^='map-']");
  selects.forEach(s => {
    const stdField = s.id.replace("map-", "");
    if (s.value) mapping[stdField] = s.value;
  });

  try {
    const res = await fetch("/api/map-columns", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: currentDatasetId,
        mapping: mapping
      })
    });

    if (!res.ok) throw new Error("Mapping failed.");
    const data = await res.json();

    closeColumnMapperModal();
    renderValidationReport(data.validation_report, 84, data.cleaned_columns.length);
    showToast("Column mapping applied successfully!", "success");
  } catch (err) {
    showToast(err.message, "error");
  }
}

// Run Inference on Uploaded Dataset
async function runInferenceOnCurrentDataset() {
  showToast("Running XGBoost & SHAP Inference Engine...", "info");
  try {
    const formData = new FormData();
    formData.append("dataset_id", currentDatasetId);

    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Inference execution failed.");
    const data = await res.json();

    currentPredictionId = data.prediction_id;
    await loadInitialPredictions();
    showToast("Early warning predictions updated!", "success");
    switchTab("portfolio");
  } catch (err) {
    showToast(err.message, "error");
  }
}

// Load Pre-configured 84-Field Demo Dataset
async function loadDemoDataset() {
  currentPredictionId = "demo-84-fields";
  showToast("Loading 84-field demonstration dataset...", "info");
  await loadInitialPredictions();
  showToast("Demo dataset loaded successfully!", "success");
}

// Download Prediction CSV
function exportPredictionsCSV() {
  window.open(`/api/export/${currentPredictionId}`, "_blank");
}

// Download Sample Template CSV
function downloadTemplateCSV() {
  window.open("/api/sample-csv?type=prediction", "_blank");
}

// Load MLOps Metrics & Benchmark Table
async function loadMLOpsMetrics() {
  try {
    const res = await fetch("/api/model/metrics");
    if (!res.ok) return;
    const metrics = await res.json();
    if (!metrics.yield_metrics) return;

    // KPI Cards
    document.getElementById("ml-mae").textContent = `${metrics.yield_metrics.mae} t/ac`;
    document.getElementById("ml-r2").textContent = metrics.yield_metrics.r2;
    document.getElementById("ml-roc-auc").textContent = metrics.risk_metrics.roc_auc;
    document.getElementById("ml-capture-rate").textContent = `${metrics.risk_metrics.business_top20_capture_rate}%`;

    // Comparison Table
    const tbody = document.getElementById("model-comparison-table-body");
    tbody.innerHTML = metrics.model_comparison.map(m => `
      <tr>
        <td style="font-weight: 700; color: ${m.model.includes('Production') ? 'var(--emerald-light)' : 'var(--text-main)'};">
          ${m.model}
        </td>
        <td>${m.mae}</td>
        <td>${m.rmse}</td>
        <td style="font-weight: 600;">${m.r2}</td>
        <td>${m.training_time}</td>
        <td>
          <span class="badge ${m.model.includes('Production') ? 'badge-low' : 'badge-medium'}">
            ${m.model.includes('Production') ? 'ACTIVE / PROD' : 'BENCHMARK'}
          </span>
        </td>
      </tr>
    `).join("");

    // Feature Importance Chart
    renderFeatureImportanceChart(metrics.top_features);

    // Confusion Matrix
    renderConfusionMatrix(metrics.risk_metrics.confusion_matrix);
  } catch (err) {
    console.error("MLOps load error:", err);
  }
}

// Render Feature Importance Horizontal Bar Chart
function renderFeatureImportanceChart(features) {
  const ctx = document.getElementById("chart-feature-importance").getContext("2d");
  if (chartFeatureImp) chartFeatureImp.destroy();

  const labels = features.map(f => f.feature.replace(/_/g, " "));
  const values = features.map(f => f.importance);

  chartFeatureImp = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Feature Importance (Gini Gain)",
        data: values,
        backgroundColor: "rgba(52, 211, 153, 0.7)",
        borderColor: "#34d399",
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(52, 211, 153, 0.08)" } },
        y: { ticks: { color: "#94a3b8", font: { size: 11 } }, grid: { display: false } }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// Render Confusion Matrix
function renderConfusionMatrix(cm) {
  const container = document.getElementById("confusion-matrix-display");
  if (!cm || cm.length < 2) return;

  const [tn, fp] = cm[0];
  const [fn, tp] = cm[1];

  container.innerHTML = `
    <div style="display: grid; grid-template-columns: 100px 1fr 1fr; gap: 8px; text-align: center; font-size: 12px;">
      <div></div>
      <div style="font-weight: 700; color: var(--text-dim);">PRED NORMAL</div>
      <div style="font-weight: 700; color: var(--coral-light);">PRED FAILURE</div>

      <div style="font-weight: 700; color: var(--text-dim); text-align: right; padding-top: 18px;">ACTUAL NORMAL</div>
      <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid var(--emerald-dark); border-radius: var(--radius-md); padding: 16px;">
        <div style="font-size: 20px; font-weight: 800; color: var(--emerald-light);">${tn}</div>
        <span style="font-size: 11px; color: var(--text-muted);">True Negative</span>
      </div>
      <div style="background: var(--bg-input); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 16px;">
        <div style="font-size: 20px; font-weight: 800; color: var(--amber-light);">${fp}</div>
        <span style="font-size: 11px; color: var(--text-muted);">False Positive</span>
      </div>

      <div style="font-weight: 700; color: var(--coral-light); text-align: right; padding-top: 18px;">ACTUAL FAILURE</div>
      <div style="background: var(--bg-input); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 16px;">
        <div style="font-size: 20px; font-weight: 800; color: var(--coral-light);">${fn}</div>
        <span style="font-size: 11px; color: var(--text-muted);">False Negative</span>
      </div>
      <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid var(--coral-danger); border-radius: var(--radius-md); padding: 16px;">
        <div style="font-size: 20px; font-weight: 800; color: var(--coral-light);">${tp}</div>
        <span style="font-size: 11px; color: var(--text-muted);">True Positive</span>
      </div>
    </div>
  `;
}

// Trigger Model Retrain
async function triggerModelRetrain() {
  showToast("Triggering leakage-safe model retraining on historical harvest data...", "info");
  try {
    const res = await fetch("/api/train", { method: "POST" });
    if (!res.ok) throw new Error("Retraining failed.");
    const data = await res.json();
    showToast("Model retrained successfully!", "success");
    await loadMLOpsMetrics();
    await loadInitialPredictions();
  } catch (err) {
    showToast(err.message, "error");
  }
}

// Dark/Light Theme Toggle
function toggleTheme() {
  document.body.classList.toggle("light-theme");
  const isLight = document.body.classList.contains("light-theme");
  const themeIcon = document.getElementById("theme-icon");
  if (themeIcon) {
    themeIcon.setAttribute("data-lucide", isLight ? "moon" : "sun-moon");
    if (window.lucide) lucide.createIcons();
  }
}

// ==========================================
// INGESTED RAW DATA EXPLORER
// ==========================================

// Load Raw Ingested Data from API
async function loadRawDatasetData() {
  try {
    const res = await fetch(`/api/dataset/${encodeURIComponent(currentDatasetId)}/data`);
    if (!res.ok) return;
    const json = await res.json();
    
    rawDatasetRecords = json.data || [];
    rawDatasetStats = json.stats || {};
    
    renderRawDataStats();
    renderRawDataTable();
  } catch (err) {
    console.error("Failed to load raw dataset:", err);
  }
}

// Render Summary Stats Cards for Raw Dataset
function renderRawDataStats() {
  const s = rawDatasetStats;
  
  if (s.soil_moisture) {
    document.getElementById("raw-stat-moisture").textContent = `${s.soil_moisture.mean}%`;
    document.getElementById("raw-range-moisture").textContent = `Range: ${s.soil_moisture.min}% to ${s.soil_moisture.max}%`;
  }
  
  if (s.temperature) {
    document.getElementById("raw-stat-temp").textContent = `${s.temperature.mean} °C`;
    document.getElementById("raw-range-temp").textContent = `Range: ${s.temperature.min} to ${s.temperature.max} °C`;
  }
  
  if (s.ndvi) {
    document.getElementById("raw-stat-ndvi").textContent = `${s.ndvi.mean}`;
    document.getElementById("raw-range-ndvi").textContent = `Range: ${s.ndvi.min} to ${s.ndvi.max}`;
  }
  
  if (s.disease_score) {
    document.getElementById("raw-stat-disease").textContent = `${s.disease_score.mean} / 100`;
    document.getElementById("raw-range-disease").textContent = `Range: ${s.disease_score.min} to ${s.disease_score.max}`;
  }
}

// Render Ingested Data Table Grid
function renderRawDataTable() {
  const tbody = document.getElementById("raw-data-table-body");
  if (!tbody) return;
  
  const search = (document.getElementById("raw-data-search")?.value || "").toLowerCase();
  const cropFilter = document.getElementById("raw-crop-filter")?.value || "ALL";

  const filtered = rawDatasetRecords.filter(row => {
    const matchSearch = !search ||
      String(row.field_id || "").toLowerCase().includes(search) ||
      String(row.crop_type || "").toLowerCase().includes(search) ||
      String(row.variety || "").toLowerCase().includes(search) ||
      String(row.growth_stage || "").toLowerCase().includes(search);
      
    const matchCrop = cropFilter === "ALL" || String(row.crop_type || "").toLowerCase() === cropFilter.toLowerCase();
    return matchSearch && matchCrop;
  });

  const counter = document.getElementById("raw-record-counter");
  if (counter) {
    counter.textContent = `Showing ${filtered.length} of ${rawDatasetRecords.length} observations`;
  }

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="18" style="text-align: center; color: var(--text-muted); padding: 24px;">No matching observations found.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(r => `
    <tr>
      <td style="font-weight: 700; color: var(--emerald-light);">${r.field_id || '--'}</td>
      <td style="font-size: 11px; color: var(--text-dim);">${r.date ? String(r.date).split(' ')[0] : '--'}</td>
      <td><strong>${r.crop_type || '--'}</strong></td>
      <td style="font-size: 11px; color: var(--text-dim);">${r.variety || '--'}</td>
      <td>${r.crop_age_days ?? '--'}</td>
      <td><span style="font-size: 12px; color: var(--text-muted);">${r.growth_stage || '--'}</span></td>
      <td style="font-weight: 600; color: ${(r.soil_moisture < 30) ? 'var(--coral-light)' : 'var(--emerald-light)'};">${r.soil_moisture ?? '--'}%</td>
      <td>${r.soil_temperature ?? '--'}</td>
      <td>${r.soil_ph ?? '--'}</td>
      <td>${r.nitrogen ?? '--'}</td>
      <td>${r.phosphorus ?? '--'}</td>
      <td>${r.potassium ?? '--'}</td>
      <td>${r.temperature ?? '--'}</td>
      <td>${r.humidity ?? '--'}%</td>
      <td>${r.rainfall ?? '--'}</td>
      <td style="font-weight: 600; color: ${(r.ndvi < 0.6) ? 'var(--coral-light)' : 'var(--emerald-light)'};">${r.ndvi ?? '--'}</td>
      <td style="color: ${(r.disease_score > 25) ? 'var(--coral-light)' : 'var(--text-main)'}; font-weight: 600;">${r.disease_score ?? '--'}</td>
      <td>${r.pest_count ?? '--'}</td>
    </tr>
  `).join("");
}

function filterRawDataTable() {
  renderRawDataTable();
}
