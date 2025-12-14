document.addEventListener("DOMContentLoaded", () => {
  const imageInput = document.getElementById("image-input");
  const previewImage = document.getElementById("preview-image");
  const previewText = document.querySelector(".image-preview-text");
  const predictButton = document.getElementById("predict-button");
  const resultText = document.getElementById("result-text");
  const exampleImage = document.getElementById("example-image");

  const HF_API_BASE =
    "https://ispe-age-predictor.hf.space/gradio_api/call/predict_1";
  let imageDataUrl = null;

  /**
   * Reads an image file or blob and displays it in the preview.
   * @param {File|Blob} source - The image source to display.
   */
  function displayImage(source) {
    const reader = new FileReader();
    reader.onload = (e) => {
      imageDataUrl = e.target.result;
      previewImage.src = imageDataUrl;
      previewImage.style.display = "block";
      previewText.style.display = "none";
    };
    reader.readAsDataURL(source);
  }

  // Listener for user-uploaded files.
  imageInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) displayImage(file);
  });

  // Listener for the clickable example image.
  exampleImage.addEventListener("click", async () => {
    try {
      const blob = await fetch(exampleImage.src).then((res) => res.blob());
      displayImage(blob);
    } catch (err) {
      console.error("Error loading example image:", err);
      alert("Failed to load the example image.");
    }
  });

  // parse SSE-style text
  function parseSSE(text) {
    const blocks = text
      .split(/\n\n+/)
      .map((b) => b.trim())
      .filter(Boolean);
    const parsedEvents = [];
    for (const block of blocks) {
      const lines = block.split(/\n/).map((l) => l.trim());
      let eventType = null;
      const dataLines = [];
      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventType = line.slice("event:".length).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice("data:".length).trim());
        }
      }
      if (dataLines.length) {
        const combined = dataLines.join("\n");
        try {
          const parsed = JSON.parse(combined);
          parsedEvents.push({ eventType, data: parsed });
        } catch (err) {
          // ignore non-JSON blocks
        }
      }
    }
    return parsedEvents;
  }

  // Poll the event_id endpoint.
  async function pollForResult(eventId) {
    try {
      const resp = await fetch(`${HF_API_BASE}/${eventId}`);
      const text = await resp.text();
      const events = parseSSE(text);

      if (!events.length) throw new Error("No JSON data in SSE response");

      const last = events[events.length - 1];
      const data = last.data;

      if (Array.isArray(data)) {
        resultText.textContent = String(data[0]);
        predictButton.disabled = false;
        return;
      }

      if (data?.status) {
        if (data.status === "COMPLETE") {
          const out =
            (data.output && Array.isArray(data.output) && data.output[0]) ||
            data.output?.data?.[0] ||
            (data.data && Array.isArray(data.data) && data.data[0]) ||
            null;
          resultText.textContent =
            out !== null ? String(out) : JSON.stringify(data);
          predictButton.disabled = false;
          return;
        } else if (data.status === "PROCESSING" || data.status === "QUEUED") {
          resultText.textContent = `Predicting... (Status: ${data.status})`;
          setTimeout(() => pollForResult(eventId), 1000);
          return;
        } else {
          throw new Error(`Job failed with status: ${data.status}`);
        }
      }

      resultText.textContent = JSON.stringify(data);
      predictButton.disabled = false;
    } catch (err) {
      console.error("Polling error:", err);
      resultText.textContent = "Prediction failed while polling. See console.";
      predictButton.disabled = false;
    }
  }

  // Start prediction.
  predictButton.addEventListener("click", async () => {
    if (!imageDataUrl) {
      alert("Please choose an image first!");
      return;
    }
    predictButton.disabled = true;
    resultText.textContent = "Sending to Hugging Face...";

    try {
      const payload = { data: [imageDataUrl] };
      const resp = await fetch(HF_API_BASE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const txt = await resp.text().catch(() => "");
        throw new Error(`HTTP ${resp.status} ${resp.statusText} ${txt}`);
      }

      const j = await resp.json();
      const eventId = j?.event_id || j?.id;
      if (!eventId) throw new Error("No event_id returned from API");

      resultText.textContent = "Waiting for prediction (polling)...";
      pollForResult(eventId);
    } catch (err) {
      console.error("Prediction error:", err);
      resultText.textContent = "Prediction failed: " + (err.message || err);
      predictButton.disabled = false;
    }
  });
});
