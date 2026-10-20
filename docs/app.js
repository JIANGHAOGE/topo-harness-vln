(function () {
  "use strict";
  var fallback = {
    status: "planned",
    updated: "2026-10-20",
    metrics: [
      { label: "Completed runs", value: "0" },
      { label: "Planned ablations", value: "5" },
      { label: "Reported metrics", value: "0" },
      { label: "Demo episodes", value: "0" }
    ],
    ablations: [
      { variant: "Full Topo-Harness", success: "TBD", efficiency: "TBD", time: "TBD", nodes: "TBD", falseFinish: "TBD" },
      { variant: "No semantic topology", success: "TBD", efficiency: "TBD", time: "TBD", nodes: "TBD", falseFinish: "TBD" },
      { variant: "No joint ranking", success: "TBD", efficiency: "TBD", time: "TBD", nodes: "TBD", falseFinish: "TBD" },
      { variant: "No evolving memory", success: "TBD", efficiency: "TBD", time: "TBD", nodes: "TBD", falseFinish: "TBD" },
      { variant: "Single-level verification", success: "TBD", efficiency: "TBD", time: "TBD", nodes: "TBD", falseFinish: "TBD" }
    ],
    demos: [
      { title: "Semantic-node exploration", description: "Planned: ranked node traversal followed by local object search.", video: "" },
      { title: "Verifier-guided recovery", description: "Planned: a failed grounding or waypoint followed by bounded recovery.", video: "" },
      { title: "Repeated-task memory", description: "Planned: compare first search with a later search using stored evidence.", video: "" }
    ]
  };

  function text(value) {
    return String(value == null ? "TBD" : value);
  }

  function render(data) {
    var status = text(data.status || "planned");
    document.getElementById("experiment-status").textContent =
      status === "complete" ? "Experiments complete" : status === "running" ? "Experiments running" : "Awaiting experiments";
    document.getElementById("hero-status").textContent = status.charAt(0).toUpperCase() + status.slice(1);
    document.getElementById("last-updated").textContent = "Last updated: " + text(data.updated);

    var metricGrid = document.getElementById("metric-grid");
    metricGrid.innerHTML = "";
    (data.metrics || []).forEach(function (metric) {
      var card = document.createElement("article");
      card.className = "metric";
      var value = document.createElement("div");
      value.className = "value";
      value.textContent = text(metric.value);
      var label = document.createElement("div");
      label.className = "label";
      label.textContent = text(metric.label);
      card.appendChild(value);
      card.appendChild(label);
      metricGrid.appendChild(card);
    });

    var body = document.getElementById("ablation-body");
    body.innerHTML = "";
    (data.ablations || []).forEach(function (row) {
      var tr = document.createElement("tr");
      ["variant", "success", "efficiency", "time", "nodes", "falseFinish"].forEach(function (key) {
        var td = document.createElement("td");
        td.textContent = text(row[key]);
        if (td.textContent === "TBD") td.className = "pending";
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });

    var demoGrid = document.getElementById("demo-grid");
    demoGrid.innerHTML = "";
    (data.demos || []).forEach(function (demo) {
      var card = document.createElement("article");
      card.className = "demo-card";
      if (demo.video) {
        var video = document.createElement("video");
        video.controls = true;
        video.preload = "metadata";
        video.src = demo.video;
        card.appendChild(video);
      } else {
        var placeholder = document.createElement("div");
        placeholder.className = "video-placeholder";
        placeholder.textContent = "Episode video pending";
        card.appendChild(placeholder);
      }
      var copy = document.createElement("div");
      copy.className = "demo-copy";
      var heading = document.createElement("h3");
      heading.textContent = text(demo.title);
      var paragraph = document.createElement("p");
      paragraph.textContent = text(demo.description);
      copy.appendChild(heading);
      copy.appendChild(paragraph);
      card.appendChild(copy);
      demoGrid.appendChild(card);
    });
  }

  fetch("./data/results.json")
    .then(function (response) {
      if (!response.ok) throw new Error("Results file unavailable");
      return response.json();
    })
    .then(render)
    .catch(function () { render(fallback); });
})();
