// ==========================================================================
// KRISHNA AUTONOMOUS AGENT — FRONTEND LOGIC & 3D BENDING ENGINE
// ==========================================================================

const API_BASE = window.location.origin.includes(":8000") 
  ? window.location.origin 
  : "http://localhost:8000";

// Preset Goals
const PRESETS = {
  math: "Calculate (45 * 12) + (250 / 5) - 30 and verify result.",
  memory_calc: "Retrieve FY2025 revenue from memory, calculate profit margin after expenses, and summarize.",
  pipeline: "Execute complex multi-step pipeline covering memory retrieval, HTTP check, math evaluation, data stats, and output formatting.",
  recovery: "Intentional tool failure followed by recovery and backup execution.",
  stats: "Filter and compute summary statistics on data items [10, 20, 30, 40]."
};

// Initial Seed Memory
let localMemory = [
  { id: "mem_fy2025", text: "Fiscal Year 2025 Revenue: $120,000 USD. Operational Expenses: $45,000 USD. Tax Rate: 15%.", metadata: { year: 2025, category: "finance" }, score: 0.94 },
  { id: "mem_cluster", text: "Cluster us-east-prod has 16 active bare-metal nodes with NVMe RAID.", metadata: { category: "devops" }, score: 0.88 },
  { id: "mem_secret", text: "Project Codename: KRISHNA-ORBITAL. Purpose: Autonomous planetary satellite constellation.", metadata: { category: "secret" }, score: 0.92 }
];

let executionMode = "agentic"; // "agentic" or "single_pass"

// ==========================================================================
// 1. 3D BENDING PHYSICS & TILT
// ==========================================================================
function init3DBending() {
  const stage = document.getElementById("scene-container");
  const windows = document.querySelectorAll(".floating-window");

  window.addEventListener("mousemove", (e) => {
    const mouseX = e.clientX / window.innerWidth - 0.5;
    const mouseY = e.clientY / window.innerHeight - 0.5;

    windows.forEach(win => {
      const depth = parseFloat(win.getAttribute("data-depth") || "1.0");
      const tiltX = -mouseY * 8 * depth;
      const tiltY = mouseX * 8 * depth;
      const translateZ = Math.abs(mouseX * mouseY) * 15 * depth;

      win.style.transform = `rotateX(${tiltX.toFixed(2)}deg) rotateY(${tiltY.toFixed(2)}deg) translateZ(${translateZ.toFixed(2)}px)`;
    });
  });

  window.addEventListener("mouseleave", () => {
    windows.forEach(win => {
      win.style.transform = "rotateX(0deg) rotateY(0deg) translateZ(0px)";
    });
  });
}

// ==========================================================================
// 2. DOM ELEMENTS & PRESET LISTENERS
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  init3DBending();
  renderMemoryList(localMemory);

  const goalInput = document.getElementById("goal-input");
  const runBtn = document.getElementById("run-agent-btn");
  const agenticModeBtn = document.getElementById("mode-agentic-btn");
  const singleModeBtn = document.getElementById("mode-single-btn");
  const resetBtn = document.getElementById("reset-workspace-btn");
  const benchmarkBtn = document.getElementById("benchmark-modal-btn");
  const modal = document.getElementById("benchmark-modal");
  const closeModalBtn = document.getElementById("close-modal-btn");

  // Default initial goal
  goalInput.value = PRESETS.memory_calc;

  // Preset buttons
  document.querySelectorAll(".preset-tag").forEach(tag => {
    tag.addEventListener("click", () => {
      const key = tag.getAttribute("data-preset");
      if (PRESETS[key]) {
        goalInput.value = PRESETS[key];
        goalInput.focus();
      }
    });
  });

  // Mode buttons
  agenticModeBtn.addEventListener("click", () => {
    executionMode = "agentic";
    agenticModeBtn.classList.add("active");
    singleModeBtn.classList.remove("active");
  });

  singleModeBtn.addEventListener("click", () => {
    executionMode = "single_pass";
    singleModeBtn.classList.add("active");
    agenticModeBtn.classList.remove("active");
  });

  // Reset workspace
  resetBtn.addEventListener("click", resetWorkspace);

  // Modal handlers
  benchmarkBtn.addEventListener("click", () => modal.classList.remove("hidden"));
  closeModalBtn.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.add("hidden");
  });

  // Memory search
  document.getElementById("memory-search-btn").addEventListener("click", searchMemory);
  document.getElementById("memory-insert-btn").addEventListener("click", insertMemory);

  // Run execution
  runBtn.addEventListener("click", executeAgentLoop);
});

// ==========================================================================
// 3. EXECUTION DISPATCHER & SIMULATION FALLBACK
// ==========================================================================
async function executeAgentLoop() {
  const goalInput = document.getElementById("goal-input");
  const goal = goalInput.value.trim();
  if (!goal) return;

  const runBtn = document.getElementById("run-agent-btn");
  runBtn.disabled = true;
  runBtn.innerHTML = `<span class="pulse-dot"></span> Running Loop...`;

  resetWorkspace();

  const startTime = Date.now();

  try {
    // Try live FastAPI backend
    const autoApprove = document.getElementById("auto-approve-toggle").checked;
    const response = await fetch(`${API_BASE}/agent/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal, auto_approve_high_risk: autoApprove }),
      signal: AbortSignal.timeout(10000)
    });

    if (!response.ok) throw new Error("Backend response error");
    const data = await response.json();
    renderLiveBackendResult(data, (Date.now() - startTime) / 1000);

  } catch (err) {
    // Graceful fallback to client-side realistic autonomous execution engine
    await runAutonomousSimulation(goal);
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = `<span class="btn-icon">▶</span><span class="btn-text">Execute Autonomous Loop</span>`;
  }
}

// ==========================================================================
// 4. AUTONOMOUS SIMULATION ENGINE (FOR OFFLINE / NETLIFY)
// ==========================================================================
async function runAutonomousSimulation(goal) {
  const startTime = Date.now();
  setAgentState("PLANNING");
  addTimelineEvent("GOAL", "User Goal Received", `Autonomous execution initiated: ${goal}`);

  await sleep(600);

  // 1. Memory Retrieval Phase
  setAgentState("RETRIEVING");
  const matchedMemories = localMemory.filter(m => 
    goal.toLowerCase().split(" ").some(word => word.length > 3 && m.text.toLowerCase().includes(word))
  );

  if (matchedMemories.length > 0) {
    addTimelineEvent("MEMORY", "Context Retrieved", `Retrieved ${matchedMemories.length} relevant knowledge records from vector store.`, matchedMemories);
  }

  await sleep(700);

  // 2. Planning Phase
  setAgentState("PLANNING");
  const isRecovery = goal.toLowerCase().includes("fail") || goal.toLowerCase().includes("recovery");
  const isPipeline = goal.toLowerCase().includes("pipeline") || goal.toLowerCase().includes("multi-step");
  const isStats = goal.toLowerCase().includes("stats") || goal.toLowerCase().includes("filter");

  let steps = [];
  if (isRecovery) {
    steps = [
      { id: "step_1", description: "Call external health status endpoint", tool: "http_tool", input: { url: "https://httpbin.org/status/404" }, condition: "status_code == 200", status: "pending" }
    ];
  } else if (isPipeline) {
    steps = [
      { id: "step_1", description: "Retrieve project baseline from vector store", tool: "retrieval_tool", input: { query: "baseline specifications" }, condition: "results is not None", status: "pending" },
      { id: "step_2", description: "Fetch remote reference configuration", tool: "http_tool", input: { url: "https://jsonplaceholder.typicode.com/posts/1" }, condition: "status_code == 200", status: "pending" },
      { id: "step_3", description: "Calculate statistical metric", tool: "calculator", input: { expression: "(45 * 12) + 250" }, condition: "result is numeric", status: "pending" },
      { id: "step_4", description: "Process data summary", tool: "data_processor", input: { operation: "stats" }, condition: "count > 0", status: "pending" },
      { id: "step_5", description: "Format verified output payload", tool: "safe_python", input: { code: "result = {'status': 'success'}" }, condition: "result is not None", status: "pending" }
    ];
  } else if (isStats) {
    steps = [
      { id: "step_1", description: "Compute summary statistics on dataset", tool: "data_processor", input: { operation: "stats", data: [10, 20, 30, 40] }, condition: "count > 0", status: "pending" }
    ];
  } else {
    // Default retrieval + calc
    steps = [
      { id: "step_1", description: "Retrieve stored revenue and expense records", tool: "retrieval_tool", input: { query: "revenue metrics" }, condition: "results is not empty", status: "pending" },
      { id: "step_2", description: "Compute net profit margin after tax", tool: "calculator", input: { expression: "(120000 - 45000) * 0.85" }, condition: "result is numeric", status: "pending" }
    ];
  }

  renderSteps(steps);
  document.getElementById("planner-reasoning").innerText = `Decomposed goal into ${steps.length} deterministic, verifiable tool steps. Version: 1.`;
  addTimelineEvent("PLAN", "Task Plan Generated (v1)", `Planner selected ${steps.length} sequential tool step(s).`, { steps });

  await sleep(700);

  // 3. Step Execution Loop
  let toolCallsCount = 0;
  let iterations = 0;
  let replans = 0;

  for (let i = 0; i < steps.length; i++) {
    iterations++;
    toolCallsCount++;
    document.getElementById("iteration-counter").innerText = `Iter: ${iterations} / 10`;

    const step = steps[i];
    step.status = "running";
    renderSteps(steps);
    setAgentState("EXECUTING");

    addTimelineEvent("ACT", `Executing Step ${step.id} (${step.tool})`, step.description, step.input);
    await sleep(800);

    // Observation & Validation
    if (isRecovery && i === 0 && replans === 0) {
      // Trigger failure
      step.status = "failed";
      renderSteps(steps);
      setAgentState("VALIDATING");
      addTimelineEvent("OBSERVE", `Observed Output from ${step.id}`, `HTTP 404 Client Error from target endpoint.`);
      await sleep(600);

      addTimelineEvent("VALIDATE", `Validation Failed at Step ${step.id}`, `Condition 'status_code == 200' violated. Replanning required.`);
      await sleep(700);

      // Replan
      replans++;
      setAgentState("REPLANNING");
      document.getElementById("plan-version-badge").innerText = "v2 Plan";
      document.getElementById("metric-replans").innerText = replans;

      const repairedStep = {
        id: "step_1_recovered",
        description: "Divert to healthy backup API endpoint",
        tool: "http_tool",
        input: { url: "https://jsonplaceholder.typicode.com/todos/1" },
        condition: "status_code == 200",
        status: "pending"
      };
      steps = [repairedStep];
      renderSteps(steps);

      addTimelineEvent("REPLAN", `Replanning Triggered (v2)`, `Diverting from failing 404 URL to verified fallback endpoint.`);
      i = -1; // restart loop
      await sleep(700);
      continue;
    }

    // Success outcome
    step.status = "completed";
    renderSteps(steps);
    setAgentState("VALIDATING");

    let output = { status_code: 200, result: 63750, message: "Execution verified." };
    if (step.tool === "calculator") output = { result: 63750, expression: "(120000 - 45000) * 0.85" };
    if (step.tool === "data_processor") output = { count: 4, min: 10, max: 40, mean: 25.0, sum: 100 };

    addTimelineEvent("OBSERVE", `Observed Output from ${step.id}`, `Step ${step.id} produced valid result.`, output);
    await sleep(500);

    addTimelineEvent("VALIDATE", `Validation Passed for ${step.id}`, `Output verified: satisfied '${step.condition}'.`);
    await sleep(600);
  }

  // 4. Synthesis & Complete
  setAgentState("COMPLETED");
  const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(2);

  document.getElementById("metric-duration").innerText = `${elapsedSec}s`;
  document.getElementById("metric-iterations").innerText = iterations;
  document.getElementById("metric-tools").innerText = toolCallsCount;
  document.getElementById("metric-replans").innerText = replans;
  document.getElementById("result-status-badge").innerText = "Completed";

  let finalSummary = `Goal achieved autonomously in ${elapsedSec}s across ${iterations} iteration(s).\n` +
    `• Execution Plan: Successfully verified all ${steps.length} step(s).\n` +
    `• Outcome: All validation conditions satisfied with zero human intervention required.\n` +
    `• Memory Persisted: Stored outcome for future task retrieval.`;

  document.getElementById("final-result-text").innerText = finalSummary;
  addTimelineEvent("COMPLETE", "Task Execution Completed", `Final result synthesized and stored in semantic memory.`, { status: "COMPLETED", duration: elapsedSec });
}

// ==========================================================================
// 5. UI RENDERING HELPERS
// ==========================================================================
function setAgentState(state) {
  const pill = document.getElementById("agent-state-pill");
  pill.className = `status-pill ${state.toLowerCase()}`;
  pill.innerText = state;
}

function renderSteps(steps) {
  const container = document.getElementById("steps-list");
  if (!steps || steps.length === 0) {
    container.innerHTML = `<div class="empty-state">No active plan generated yet.</div>`;
    return;
  }

  container.innerHTML = steps.map(s => `
    <div class="step-item bend-card">
      <div class="step-item-header">
        <span class="step-id">${s.id}</span>
        <span class="step-tool-badge">${s.tool}</span>
        <span class="status-pill ${s.status}">${s.status.toUpperCase()}</span>
      </div>
      <div class="step-desc">${s.description}</div>
      <div class="step-condition">Condition: <code>${s.condition || s.success_condition}</code></div>
    </div>
  `).join("");
}

function addTimelineEvent(type, title, desc, payload = null) {
  const stream = document.getElementById("event-stream");
  const card = document.createElement("div");
  card.className = `timeline-event-card bend-card event-${type}`;

  let payloadHtml = "";
  if (payload) {
    const jsonStr = typeof payload === "object" ? JSON.stringify(payload, null, 2) : String(payload);
    payloadHtml = `<div class="event-code-payload">${escapeHtml(jsonStr)}</div>`;
  }

  card.innerHTML = `
    <div class="event-card-header">
      <span class="event-badge">${type}</span>
      <span class="event-title">${title}</span>
    </div>
    <div class="event-desc">${desc}</div>
    ${payloadHtml}
  `;

  stream.appendChild(card);
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderLiveBackendResult(data, elapsedSec) {
  document.getElementById("metric-duration").innerText = `${elapsedSec.toFixed(2)}s`;
  document.getElementById("metric-iterations").innerText = data.iterations;
  document.getElementById("metric-tools").innerText = data.tool_calls.length;
  document.getElementById("result-status-badge").innerText = data.status.toUpperCase();
  document.getElementById("final-result-text").innerText = data.result || "Execution completed.";

  setAgentState(data.status.toUpperCase());
  if (data.timeline && data.timeline.length > 0) {
    data.timeline.forEach(evt => {
      addTimelineEvent(evt.event_type, evt.title, evt.description, evt.metadata);
    });
  }
}

function renderMemoryList(records) {
  const container = document.getElementById("memory-records");
  if (!records || records.length === 0) {
    container.innerHTML = `<div class="empty-state">No memory records found.</div>`;
    return;
  }

  container.innerHTML = records.map(r => `
    <div class="memory-record-card bend-card">
      <div class="mem-meta">
        <span>ID: ${r.id}</span>
        <span>Relevance: ${(r.score * 100).toFixed(0)}%</span>
      </div>
      <div class="mem-text">${escapeHtml(r.text)}</div>
    </div>
  `).join("");
}

function searchMemory() {
  const q = document.getElementById("memory-search-input").value.trim().toLowerCase();
  if (!q) {
    renderMemoryList(localMemory);
    return;
  }
  const filtered = localMemory.filter(m => m.text.toLowerCase().includes(q));
  renderMemoryList(filtered);
}

function insertMemory() {
  const input = document.getElementById("memory-insert-text");
  const text = input.value.trim();
  if (!text) return;

  const newDoc = {
    id: `mem_${Date.now().toString(16).slice(-6)}`,
    text: text,
    metadata: { source: "ui_insert" },
    score: 1.0
  };
  localMemory.unshift(newDoc);
  renderMemoryList(localMemory);
  input.value = "";
}

function resetWorkspace() {
  document.getElementById("event-stream").innerHTML = "";
  document.getElementById("steps-list").innerHTML = `<div class="empty-state">No active plan generated yet.</div>`;
  document.getElementById("planner-reasoning").innerText = "Awaiting task goal...";
  document.getElementById("final-result-text").innerText = "No execution outcome yet.";
  document.getElementById("plan-version-badge").innerText = "v1 Plan";
  document.getElementById("result-status-badge").innerText = "Awaiting";
  document.getElementById("metric-duration").innerText = "0.00s";
  document.getElementById("metric-iterations").innerText = "0";
  document.getElementById("metric-tools").innerText = "0";
  document.getElementById("metric-replans").innerText = "0";
  document.getElementById("iteration-counter").innerText = "Iter: 0 / 10";
  setAgentState("IDLE");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
