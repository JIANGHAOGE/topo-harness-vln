(function () {
  "use strict";
  var fallback = { status: "unavailable", updated: "—", metrics: [], ablations: [], demos: [] };

  function text(value) {
    return String(value == null ? "TBD" : value);
  }

  function render(data) {
    var status = text(data.status || "planned");
    document.getElementById("experiment-status").textContent =
      status === "complete" ? "250 runs reviewed · one scene" : "Results unavailable";
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
      ["variant", "valid", "invalid", "noCompletion"].forEach(function (key) {
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
      if (demo.image) {
        var imageLink = document.createElement("a");
        imageLink.href = demo.image;
        imageLink.target = "_blank";
        imageLink.rel = "noopener noreferrer";
        imageLink.className = "demo-image-link";
        var image = document.createElement("img");
        image.src = demo.image;
        image.alt = text(demo.alt);
        image.loading = "lazy";
        imageLink.appendChild(image);
        card.appendChild(imageLink);
      } else if (demo.video) {
        var video = document.createElement("video");
        video.controls = true;
        video.preload = "metadata";
        video.src = demo.video;
        card.appendChild(video);
      } else {
        var placeholder = document.createElement("div");
        placeholder.className = "video-placeholder";
        placeholder.textContent = "Media unavailable";
        card.appendChild(placeholder);
      }
      var copy = document.createElement("div");
      copy.className = "demo-copy";
      var theme = document.createElement("p");
      theme.className = "demo-theme";
      theme.textContent = text(demo.theme) + " · " + text(demo.id);
      var heading = document.createElement("h3");
      heading.textContent = text(demo.title);
      var paragraph = document.createElement("p");
      paragraph.textContent = text(demo.description);
      copy.appendChild(theme);
      copy.appendChild(heading);
      copy.appendChild(paragraph);
      if (demo.image) {
        var fullImage = document.createElement("a");
        fullImage.href = demo.image;
        fullImage.target = "_blank";
        fullImage.rel = "noopener noreferrer";
        fullImage.className = "demo-open";
        fullImage.textContent = "Open full-resolution evidence ↗";
        copy.appendChild(fullImage);
      }
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
