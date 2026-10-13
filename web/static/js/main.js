// Main client logic for Autonomous Data Analysis Agent

document.addEventListener("DOMContentLoaded", () => {
    fetchDatasetInfo();
    fetchMemories();
});

function switchTab(tabId) {
    document.querySelectorAll(".tab-pane").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
    
    document.getElementById(tabId).classList.add("active");
    event.currentTarget.classList.add("active");
}

function setPrompt(text) {
    const input = document.getElementById("query-input");
    input.value = text;
    input.focus();
}

async function fetchDatasetInfo() {
    try {
        const res = await fetch("/api/dataset-info");
        const data = await res.json();
        if (data.status === "success") {
            document.getElementById("active-dataset-name").textContent = data.name;
            document.getElementById("active-dataset-meta").textContent = `${data.rows.toLocaleString()} rows • ${data.columns} cols`;
            
            // Render Schema
            renderSchema(data.schema);
        }
    } catch (e) {
        console.error("Failed to fetch dataset info:", e);
    }
}

function renderSchema(schema) {
    const container = document.getElementById("schema-list");
    container.innerHTML = "";

    if (!schema || !schema.columns) {
        container.innerHTML = "<p class='dim-text'>No schema available</p>";
        return;
    }

    Object.entries(schema.columns).forEach(([col, info]) => {
        const row = document.createElement("div");
        row.className = "schema-row";
        row.innerHTML = `
            <div class="schema-row-top">
                <span class="col-name">${col}</span>
                <span class="col-type">${info.dtype}</span>
            </div>
            <div class="schema-row-meta">
                Nulls: ${info.null_pct}% (${info.nulls}) • Unique: ${info.uniques}
            </div>
        `;
        container.appendChild(row);
    });
}

async function loadSample(sampleName) {
    document.querySelectorAll(".pill-btn").forEach(b => b.classList.remove("active"));
    event.currentTarget.classList.add("active");

    appendMessage("user", `Switch to sample dataset: ${sampleName}`);
    try {
        const res = await fetch("/api/load-sample", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sample: sampleName })
        });
        const data = await res.json();
        if (data.status === "success") {
            appendMessage("assistant", `Loaded sample dataset **${sampleName}** (${data.rows.toLocaleString()} rows, ${data.columns} columns).`);
            fetchDatasetInfo();
        } else {
            appendMessage("assistant", `Error loading sample: ${data.message}`);
        }
    } catch (e) {
        appendMessage("assistant", `Network error loading sample: ${e}`);
    }
}

async function uploadCustomFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    appendMessage("user", `Uploading dataset: ${file.name}`);
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.status === "success") {
            appendMessage("assistant", `Uploaded and loaded **${file.name}** (${data.rows.toLocaleString()} rows, ${data.columns} columns). Ready for analysis!`);
            fetchDatasetInfo();
        } else {
            appendMessage("assistant", `Upload failed: ${data.message}`);
        }
    } catch (e) {
        appendMessage("assistant", `Upload network error: ${e}`);
    }
}

async function handleSendQuery(event) {
    event.preventDefault();
    const input = document.getElementById("query-input");
    const query = input.value.trim();
    if (!query) return;

    appendMessage("user", query);
    input.value = "";

    const sendBtn = document.getElementById("send-btn");
    sendBtn.disabled = true;
    sendBtn.innerHTML = "<span>Thinking...</span>";

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query })
        });
        const data = await res.json();

        if (data.status === "success") {
            appendAssistantResponse(data);
            fetchMemories(); // Refresh memories if any were taught or altered
        } else {
            appendMessage("assistant", `**Analysis Error:** ${data.message}`);
        }
    } catch (e) {
        appendMessage("assistant", `**Network Error:** ${e}`);
    } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = "<span>Analyze</span><span class='send-icon'>➔</span>";
    }
}

function appendMessage(role, text) {
    const container = document.getElementById("messages-container");
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}-message`;
    msgDiv.innerHTML = marked.parse(text);
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function appendAssistantResponse(data) {
    const container = document.getElementById("messages-container");
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant-message";

    let html = "";

    // Render Thought Trace Accordion if available
    if (data.thought_steps && data.thought_steps.length > 0) {
        html += `
            <div class="trace-accordion">
                <div class="trace-header" onclick="toggleTrace(this)">
                    <span>⚡ Agent Reasoning Trace (${data.thought_steps.length} steps)</span>
                    <span>▾</span>
                </div>
                <div class="trace-body">
        `;
        data.thought_steps.forEach(step => {
            html += `<div class="trace-step">`;
            html += `<div><strong>Thought:</strong> ${step.thought}</div>`;
            if (step.tool_name) {
                html += `<div class="trace-tool">Action: ${step.tool_name}</div>`;
            }
            if (step.observation) {
                html += `<div class="trace-obs">Observation: ${step.observation}</div>`;
            }
            html += `</div>`;
        });
        html += `</div></div>`;
    }

    // Render Main Markdown Answer
    html += marked.parse(data.answer);

    // Render Artifact Cards (HTML / PNG)
    if (data.artifacts && data.artifacts.length > 0) {
        data.artifacts.forEach(art => {
            if (art.endsWith(".html")) {
                html += `
                    <div class="artifact-card">
                        <span>📊 Interactive Chart: <strong>${art}</strong></span>
                        <button class="view-chart-btn" onclick="openChartModal('/artifacts/${art}')">View Chart</button>
                    </div>
                `;
            } else if (art.endsWith(".png")) {
                html += `
                    <div class="artifact-card">
                        <span>🖼️ Figure: <strong>${art}</strong></span>
                        <a href="/artifacts/${art}" target="_blank" class="view-chart-btn">Open PNG</a>
                    </div>
                `;
            }
        });
    }

    msgDiv.innerHTML = html;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function toggleTrace(header) {
    const body = header.nextElementSibling;
    const arrow = header.querySelector("span:last-child");
    if (body.style.display === "none") {
        body.style.display = "flex";
        arrow.textContent = "▾";
    } else {
        body.style.display = "none";
        arrow.textContent = "▸";
    }
}

function openChartModal(url) {
    const modal = document.getElementById("chart-modal");
    const iframe = document.getElementById("chart-iframe");
    iframe.src = url;
    modal.style.display = "flex";
}

function closeModal() {
    const modal = document.getElementById("chart-modal");
    const iframe = document.getElementById("chart-iframe");
    iframe.src = "";
    modal.style.display = "none";
}

// Memory Store Management
async function fetchMemories() {
    try {
        const res = await fetch("/api/memories");
        const data = await res.json();
        if (data.status === "success") {
            const countBadge = document.getElementById("memory-count");
            const list = document.getElementById("memory-items-list");
            countBadge.textContent = data.active_table.length;

            if (data.active_table.length === 0) {
                list.innerHTML = "<p class='dim-text'>No persistent memories saved yet.</p>";
                return;
            }

            list.innerHTML = "";
            data.active_table.forEach(m => {
                const card = document.createElement("div");
                card.className = "memory-card";
                card.innerHTML = `
                    <div class="memory-card-header">
                        <span>[${m.Category}] <strong>${m.Key}</strong> (${m.Version})</span>
                        <span style="cursor:pointer; color:#f43f5e" onclick="deleteMemory('${m.Key}')">🗑</span>
                    </div>
                    <div class="memory-desc">${m.Description}</div>
                    ${m.Expression !== "—" ? `<div class="memory-expr">${m.Expression}</div>` : ""}
                `;
                list.appendChild(card);
            });
        }
    } catch (e) {
        console.error("Failed to fetch memories:", e);
    }
}

async function saveManualMemory() {
    const key = document.getElementById("mem-key").value.trim();
    const cat = document.getElementById("mem-cat").value;
    const desc = document.getElementById("mem-desc").value.trim();
    const expr = document.getElementById("mem-expr").value.trim();

    if (!key || !desc) {
        alert("Please provide both a Key and a Description.");
        return;
    }

    try {
        const res = await fetch("/api/memories", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                key: key,
                category: cat,
                description: desc,
                formal_expression: expr || null
            })
        });
        const data = await res.json();
        if (data.status === "success") {
            document.getElementById("mem-key").value = "";
            document.getElementById("mem-desc").value = "";
            document.getElementById("mem-expr").value = "";
            fetchMemories();
        }
    } catch (e) {
        alert("Failed to save memory: " + e);
    }
}

async function deleteMemory(key) {
    if (!confirm(`Delete memory rule '${key}'?`)) return;
    try {
        await fetch(`/api/memories/${encodeURIComponent(key)}`, { method: "DELETE" });
        fetchMemories();
    } catch (e) {
        console.error(e);
    }
}

async function clearAllMemories() {
    if (!confirm("Are you sure you want to clear all persistent long-term memories?")) return;
    try {
        await fetch("/api/memories/clear", { method: "POST" });
        fetchMemories();
    } catch (e) {
        console.error(e);
    }
}
