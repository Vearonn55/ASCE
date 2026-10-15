import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import express from "express";
import { WebSocketServer } from "ws";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(__dirname, "..");
const port = Number(process.env.PORT || 8787);
const venvPython = path.join(repo, ".venv", "bin", "python");
const python = process.env.ASCE_PYTHON || (fs.existsSync(venvPython) ? venvPython : "python3");

const app = express();
app.use(express.static(path.join(__dirname, "public")));

const server = app.listen(port, "127.0.0.1", () => {
  console.log(`ASCE demo at http://127.0.0.1:${port}`);
});
server.on("error", (error) => {
  if (error.code === "EADDRINUSE") {
    console.error(
      `Port ${port} is already in use. Ctrl+Z suspends the demo without releasing the port. ` +
        `Open http://127.0.0.1:${port} and press Close, or stop the old process.`,
    );
    process.exit(1);
  }
  throw error;
});

const sockets = new WebSocketServer({ server });
const workers = new Set();

app.post("/shutdown", (_request, response) => {
  response.status(200).type("text/plain").send("closing");
  setTimeout(shutdown, 50);
});

function shutdown() {
  for (const socket of sockets.clients) socket.close();
  for (const worker of workers) worker.kill();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 500);
}

function startWorker() {
  return spawn(python, ["-m", "asce.demo_worker"], {
    cwd: repo,
    env: { ...process.env, PYTHONPATH: repo },
    stdio: ["pipe", "pipe", "inherit"],
  });
}

sockets.on("connection", (socket) => {
  const worker = startWorker();
  workers.add(worker);
  worker.on("exit", () => workers.delete(worker));
  let buffer = "";

  worker.stdout.on("data", (chunk) => {
    buffer += chunk.toString("utf8");
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line && socket.readyState === socket.OPEN) {
        socket.send(line);
      }
      newline = buffer.indexOf("\n");
    }
  });

  worker.on("exit", (code) => {
    if (socket.readyState === socket.OPEN) {
      socket.send(JSON.stringify({ type: "error", message: `worker exited (${code})` }));
      socket.close();
    }
  });

  socket.on("message", (data) => {
    const text = typeof data === "string" ? data : data.toString("utf8");
    worker.stdin.write(text.endsWith("\n") ? text : `${text}\n`);
  });

  socket.on("close", () => {
    worker.kill();
  });
});
