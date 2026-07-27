const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const { resolvePython } = require('../server/services/analysis/yolo/pythonResolver');

const MODEL_URL = 'https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf';
const MODEL_DIR = path.join(__dirname, '..', 'server', 'services', 'analysis', 'yolo', 'models');
const MODEL_PATH = path.join(MODEL_DIR, 'football-pitch-detection.pt');

function runPythonModule(python, args, options = {}) {
  const commandArgs = [...python.args, ...args];
  const result = spawnSync(python.command, commandArgs, {
    windowsHide: true,
    encoding: 'utf8',
    stdio: options.stdio || 'pipe',
  });
  return result;
}

function ensureGdownInstalled(python) {
  const check = runPythonModule(python, ['-m', 'gdown', '--version']);
  if (check.status === 0) return;

  console.log('[pitch-model] Installing gdown...');
  const install = runPythonModule(
    python,
    ['-m', 'pip', 'install', 'gdown'],
    { stdio: 'inherit' }
  );
  if (install.error || install.status !== 0) {
    throw new Error('[pitch-model] Failed to install gdown.');
  }
}

function downloadModel(python) {
  fs.mkdirSync(MODEL_DIR, { recursive: true });

  if (fs.existsSync(MODEL_PATH) && fs.statSync(MODEL_PATH).size > 0) {
    console.log(`[pitch-model] Model already exists: ${MODEL_PATH}`);
    return;
  }

  console.log(`[pitch-model] Downloading ${MODEL_URL}`);
  const download = runPythonModule(
    python,
    ['-m', 'gdown', '-O', MODEL_PATH, MODEL_URL],
    { stdio: 'inherit' }
  );
  if (download.error || download.status !== 0) {
    throw new Error('[pitch-model] Failed to download football-pitch-detection.pt.');
  }

  if (!fs.existsSync(MODEL_PATH) || fs.statSync(MODEL_PATH).size === 0) {
    throw new Error('[pitch-model] Model download finished but file is missing or empty.');
  }

  console.log(`[pitch-model] Model saved at: ${MODEL_PATH}`);
}

function main() {
  const python = resolvePython();
  console.log(`[pitch-model] Using Python: ${python.command} ${python.args.join(' ')}`.trim());
  ensureGdownInstalled(python);
  downloadModel(python);
}

try {
  main();
} catch (error) {
  console.error(error.message || error);
  process.exit(1);
}
