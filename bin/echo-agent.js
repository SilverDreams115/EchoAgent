#!/usr/bin/env node
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const cp = require("child_process");

const packageRoot = path.resolve(__dirname, "..");
const packageJson = JSON.parse(fs.readFileSync(path.join(packageRoot, "package.json"), "utf8"));
const cacheRoot = process.env.ECHO_NPM_HOME || defaultCacheRoot();
const managedVenv = path.join(cacheRoot, "venv");
const stampPath = path.join(cacheRoot, "install-stamp.json");

function defaultCacheRoot() {
  if (process.platform === "win32") {
    return path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"), "EchoAgent", "npm");
  }
  return path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache"), "echo-agent", "npm");
}

function candidatePythons() {
  const values = [];
  if (process.env.ECHO_PYTHON) values.push(process.env.ECHO_PYTHON);
  if (process.env.PYTHON) values.push(process.env.PYTHON);
  if (process.platform === "win32") {
    values.push("py -3", "python");
  } else {
    values.push("python3", "python");
  }
  return [...new Set(values)];
}

function splitCommand(command) {
  if (command.includes(" ")) {
    const parts = command.split(" ");
    return { cmd: parts[0], args: parts.slice(1) };
  }
  return { cmd: command, args: [] };
}

function runChecked(command, args, options = {}) {
  const completed = cp.spawnSync(command, args, {
    stdio: "pipe",
    encoding: "utf8",
    ...options,
  });
  return completed;
}

function findWorkingPython() {
  for (const candidate of candidatePythons()) {
    const { cmd, args } = splitCommand(candidate);
    const probe = runChecked(
      cmd,
      args.concat([
        "-c",
        "import rich, requests, typer, prompt_toolkit; print('ok')",
      ]),
      { env: process.env },
    );
    if (probe.status === 0) {
      return candidate;
    }
  }
  return null;
}

function installStampMatches() {
  try {
    const payload = JSON.parse(fs.readFileSync(stampPath, "utf8"));
    return payload.version === packageJson.version && payload.packageRoot === packageRoot;
  } catch {
    return false;
  }
}

function managedPythonPath() {
  return process.platform === "win32"
    ? path.join(managedVenv, "Scripts", "python.exe")
    : path.join(managedVenv, "bin", "python");
}

function ensureManagedInstall() {
  fs.mkdirSync(cacheRoot, { recursive: true });
  const selected = candidatePythons()[0];
  const { cmd, args } = splitCommand(selected);
  const python = findWorkingPython() || selected;
  const bootstrap = splitCommand(python);

  if (!fs.existsSync(managedPythonPath()) || !installStampMatches()) {
    let result = runChecked(bootstrap.cmd, bootstrap.args.concat(["-m", "venv", managedVenv]), { env: process.env });
    if (result.status !== 0) {
      fail(result.stderr || result.stdout || "No se pudo crear el venv administrado para EchoAgent.");
    }
    result = runChecked(managedPythonPath(), ["-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], { env: process.env });
    if (result.status !== 0) {
      fail(result.stderr || result.stdout || "No se pudo preparar pip dentro del venv administrado.");
    }
    result = runChecked(managedPythonPath(), ["-m", "pip", "install", packageRoot], { env: process.env });
    if (result.status !== 0) {
      fail(
        [
          "EchoAgent no pudo instalar sus dependencias Python automáticamente.",
          "Requisitos reales: Python 3.10+ y acceso a paquetes Python en el primer arranque, o un intérprete ya preparado con rich/requests/typer/prompt_toolkit.",
          "",
          result.stderr || result.stdout || "",
        ].join("\n"),
      );
    }
    fs.writeFileSync(stampPath, JSON.stringify({ version: packageJson.version, packageRoot }, null, 2));
  }
  return managedPythonPath();
}

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function runViaPython(pythonCommand) {
  const { cmd, args } = splitCommand(pythonCommand);
  const env = { ...process.env };
  env.PYTHONPATH = env.PYTHONPATH ? `${packageRoot}${path.delimiter}${env.PYTHONPATH}` : packageRoot;
  const child = cp.spawn(cmd, args.concat(["-m", "echo.cli.app", ...process.argv.slice(2)]), {
    stdio: "inherit",
    env,
  });
  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 1);
  });
}

const directPython = findWorkingPython();
if (directPython) {
  runViaPython(directPython);
} else {
  runViaPython(ensureManagedInstall());
}
