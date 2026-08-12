const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const { resolvePython } = require('../server/services/analysis/yolo/pythonResolver');

const MODEL_DIR = path.join(__dirname, '..', 'server', 'services', 'analysis', 'yolo', 'models');
const MODELS = [
  {
    name: 'football-player-detection.pt',
    url: 'https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q',
  },
  {
    name: 'football-pitch-detection.pt',
    url: 'https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf',
  },
  {
    name: 'football-ball-detection.pt',
    url: 'https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V',
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

function downloadModel(python, model) {
  const modelPath = path.join(MODEL_DIR, model.name);
  fs.mkdirSync(MODEL_DIR, { recursive: true });

  if (fs.existsSync(modelPath) && fs.statSync(modelPath).size > 0) {
    console.log(`[models] Model already exists: ${modelPath}`);
    return;
  }

  console.log(`[models] Downloading ${model.name} from ${model.url}`);
  const download = runPythonModule(
    python,
    ['-m', 'gdown', '-O', modelPath, model.url],
    { stdio: 'inherit' }
  );
  if (download.error || download.status !== 0) {
    throw new Error(`[models] Failed to download ${model.name}.`);
  }

  if (!fs.existsSync(modelPath) || fs.statSync(modelPath).size === 0) {
    throw new Error(`[models] ${model.name} download finished but file is missing or empty.`);
  }

  console.log(`[models] Model saved at: ${modelPath}`);
}

function main() {
  const python = resolvePython();
  console.log(`[models] Using Python: ${python.command} ${python.args.join(' ')}`.trim());
  ensureGdownInstalled(python);
  MODELS.forEach((model) => downloadModel(python, model));
}

try {
  main();
} catch (error) {
  console.error(error.message || error);
  process.exit(1);
}
