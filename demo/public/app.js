const toggle = document.querySelector("#toggle");
const closeButton = document.querySelector("#close");
const status = document.querySelector("#status");
const modeLine = document.querySelector("#mode");
const keys = document.querySelector("#keys");
const captureCount = document.querySelector("#capture-count");
const wave = document.querySelector("#wave");
const spec = document.querySelector("#spec");
const waveCtx = wave.getContext("2d");
const specCtx = spec.getContext("2d");

let audioCtx = null;
let socket = null;
let stream = null;
let raf = 0;
let pending = [];
let analyser = null;

function b64(samples) {
  const bytes = new Uint8Array(new Float32Array(samples).buffer);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 4096) {
    binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + 4096));
  }
  return btoa(binary);
}

function draw() {
  if (!analyser) return;
  const times = new Uint8Array(analyser.fftSize);
  const freqs = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(times);
  analyser.getByteFrequencyData(freqs);

  waveCtx.fillStyle = "#1c2018";
  waveCtx.fillRect(0, 0, wave.width, wave.height);
  waveCtx.beginPath();
  waveCtx.strokeStyle = "#d2f08a";
  for (let i = 0; i < times.length; i += 1) {
    const x = (i / times.length) * wave.width;
    const y = (times[i] / 255) * wave.height;
    if (i === 0) waveCtx.moveTo(x, y);
    else waveCtx.lineTo(x, y);
  }
  waveCtx.stroke();

  const column = specCtx.getImageData(1, 0, spec.width - 1, spec.height);
  specCtx.putImageData(column, 0, 0);
  const next = specCtx.createImageData(1, spec.height);
  for (let y = 0; y < spec.height; y += 1) {
    const bin = freqs[Math.floor((1 - y / spec.height) * (freqs.length - 1))];
    const index = y * 4;
    next.data[index] = 40 + bin * 0.5;
    next.data[index + 1] = 50 + bin * 0.7;
    next.data[index + 2] = 30;
    next.data[index + 3] = 255;
  }
  specCtx.putImageData(next, spec.width - 1, 0);
  raf = requestAnimationFrame(draw);
}

function clearCaptures() {
  keys.replaceChildren();
  captureCount.textContent = "0";
}

function addKey(key, confidence) {
  const el = document.createElement("div");
  el.className = "key";
  el.textContent = key;
  const sub = document.createElement("small");
  sub.textContent = confidence.toFixed(2);
  el.appendChild(sub);
  keys.appendChild(el);
  captureCount.textContent = String(keys.children.length);
  keys.scrollTop = keys.scrollHeight;
}

async function start() {
  clearCaptures();
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule("/pcm-worklet.js");
  const source = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  const worklet = new AudioWorkletNode(audioCtx, "pcm-processor");
  source.connect(worklet);
  socket = new WebSocket(`ws://${location.host}`);
  socket.addEventListener("open", () => {
    status.textContent = "Listening.";
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "ready") {
      modeLine.textContent = message.mode === "model"
        ? "Classifier checkpoint loaded."
        : "Detection only: taps are marked '?', no checkpoint loaded.";
    } else if (message.type === "predictions") {
      for (const key of message.keys) addKey(key.key, key.confidence);
    } else if (message.type === "error") {
      status.textContent = message.message;
    }
  });
  worklet.port.onmessage = (event) => {
    pending.push(...event.data);
    if (pending.length < 4096 || !socket || socket.readyState !== WebSocket.OPEN) return;
    const chunk = pending.splice(0, 4096);
    socket.send(JSON.stringify({
      type: "audio",
      sample_rate: audioCtx.sampleRate,
      pcm_b64: b64(chunk),
    }));
  };
  draw();
  toggle.textContent = "Stop";
}

async function stop() {
  cancelAnimationFrame(raf);
  if (socket) socket.close();
  if (audioCtx) await audioCtx.close();
  if (stream) stream.getTracks().forEach((track) => track.stop());
  socket = null;
  audioCtx = null;
  analyser = null;
  pending = [];
  status.textContent = "Stopped.";
  toggle.textContent = "Start";
}

toggle.addEventListener("click", async () => {
  toggle.disabled = true;
  try {
    if (audioCtx) await stop();
    else await start();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    toggle.disabled = false;
  }
});

closeButton.addEventListener("click", async () => {
  closeButton.disabled = true;
  toggle.disabled = true;
  try {
    if (audioCtx) await stop();
    status.textContent = "Closing the server...";
    await fetch("/shutdown", { method: "POST" });
    status.textContent = "Server closed. You can close this tab.";
  } catch (error) {
    status.textContent = "Server closed. You can close this tab.";
  }
});
