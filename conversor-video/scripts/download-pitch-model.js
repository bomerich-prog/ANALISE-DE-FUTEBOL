const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const { resolvePython } = require('../server/services/analysis/yolo/pythonResolver');

const MODEL_DIR = path.join(__dirname, '..', 'server', 'services', 'analysis', 'yolo', 'models');
const MODELS_TO_DOWNLOAD = [
  {
    name: 'football-player-detection.pt',
    url: 'https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q',
  },
  {
    name: 'football-pitch-detection.pt',
    url: 'https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf',
  },
];

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

  console.log('[roboflow-models] Installing gdown...');
  const install = runPythonModule(
    python,
    ['-m', 'pip', 'install', 'gdown'],
    { stdio: 'inherit' }
  );
  if (install.error || install.status !== 0) {
    throw new Error('[roboflow-models] Failed to install gdown.');
  }
}

function downloadModels(python) {
  fs.mkdirSync(MODEL_DIR, { recursive: true });

  for (const model of MODELS_TO_DOWNLOAD) {
    const modelPath = path.join(MODEL_DIR, model.name);
    if (fs.existsSync(modelPath) && fs.statSync(modelPath).size > 0) {
      console.log(`[roboflow-models] Model already exists: ${modelPath}`);
      continue;
    }

    console.log(`[roboflow-models] Downloading ${model.url}`);
    const download = runPythonModule(
      python,
      ['-m', 'gdown', '-O', modelPath, model.url],
      { stdio: 'inherit' }
    );
    if (download.error || download.status !== 0) {
      throw new Error(`[roboflow-models] Failed to download ${model.name}.`);
    }

    if (!fs.existsSync(modelPath) || fs.statSync(modelPath).size === 0) {
      throw new Error(`[roboflow-models] Download finished but file is missing or empty: ${model.name}`);
    }

    console.log(`[roboflow-models] Model saved at: ${modelPath}`);
  }
}

function main() {
  const python = resolvePython();
  console.log(`[roboflow-models] Using Python: ${python.command} ${python.args.join(' ')}`.trim());
  ensureGdownInstalled(python);
  downloadModels(python);
}

try {
  main();
} catch (error) {
  console.error(error.message || error);
  process.exit(1);
}
