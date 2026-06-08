const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const { db, dataDir } = require('../../../db');
const { AnalysisProvider, updateAnalysisProgress } = require('../AnalysisProvider');
const { resolvePythonCached } = require('./pythonResolver');

const WINDOWS_STORE_ALIAS_HINT =
  'O Windows tentou abrir o atalho da Microsoft Store para Python. Instale o Python a partir de https://www.python.org/downloads/ (marcando "Add Python to PATH") ou desabilite o alias em Configurações > Aplicativos > Configurações avançadas do aplicativo > Aliases de execução do aplicativo, e reinicie o servidor. Alternativamente, defina YOLO_PYTHON_BIN apontando para um python.exe válido (ex.: "py -3").';

function isEnabled(value) {
  return ['1', 'true', 'yes', 'on'].includes(String(value || '').toLowerCase());
}

class YoloAnalysisProvider extends AnalysisProvider {
  constructor(options = {}) {
    super();
    this.pythonBinOption = options.pythonBin || process.env.YOLO_PYTHON_BIN || null;
    this.model = options.model || process.env.YOLO_MODEL || 'yolov8n.pt';
    this.confidence = options.confidence || process.env.YOLO_CONFIDENCE || '0.25';
    this.imageSize = options.imageSize || process.env.YOLO_IMAGE_SIZE || '640';
    this.workerPath = options.workerPath || path.join(__dirname, 'run_yolo_analysis.py');
    this.debugLiveFrames =
      options.debugLiveFrames ?? isEnabled(process.env.YOLO_DEBUG_LIVE_FRAMES);
    this.debugFrameInterval =
      options.debugFrameInterval || process.env.YOLO_DEBUG_FRAME_INTERVAL || '1';
  }

  resolvePython() {
    return resolvePythonCached({ pythonBin: this.pythonBinOption });
  }

  async analyze(analysisId, videoId, userId, onProgress) {
    const video = db
      .prepare('SELECT * FROM conversions WHERE id = ? AND user_id = ?')
      .get(videoId, userId);

    if (!video) {
      throw new Error('Vídeo não encontrado para análise YOLO.');
    }

    if (!fs.existsSync(video.file_path)) {
      throw new Error('Arquivo de vídeo não encontrado no servidor.');
    }

    const outputDir = path.join(dataDir, 'analyses', String(userId), String(analysisId));
    fs.mkdirSync(outputDir, { recursive: true });
    const outputPath = path.join(outputDir, 'annotated.mp4');

    const workerArgs = [
      this.workerPath,
      '--input',
      video.file_path,
      '--output',
      outputPath,
      '--model',
      this.model,
      '--conf',
      String(this.confidence),
      '--imgsz',
      String(this.imageSize),
    ];

    if (this.debugLiveFrames) {
      workerArgs.push(
        '--debug-frame-dir',
        outputDir,
        '--debug-frame-interval',
        String(this.debugFrameInterval)
      );
    }

    return this.runWorker(workerArgs, analysisId, onProgress).then((result) => ({
      ...result,
      video_id: videoId,
      source_video_name: video.video_name,
    }));
  }

  runWorker(workerArgs, analysisId, onProgress) {
    return new Promise((resolve, reject) => {
      let python;
      try {
        python = this.resolvePython();
      } catch (err) {
        reject(err);
        return;
      }

      const spawnArgs = [...python.args, ...workerArgs];
      const child = spawn(python.command, spawnArgs, {
        cwd: path.join(__dirname, '..', '..', '..', '..'),
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
        },
        windowsHide: true,
      });

      let stdoutBuffer = '';
      let stderrBuffer = '';
      let finalResult = null;
      let workerError = null;

      const handleLine = (line) => {
        if (!line.trim()) return;
        let payload;
        try {
          payload = JSON.parse(line);
        } catch {
          stderrBuffer += `${line}\n`;
          return;
        }

        if (payload.type === 'progress') {
          const details = {
            progress_stats: {
              frames_processed: payload.framesProcessed || 0,
              players_detected: payload.playersDetected || 0,
              ball_detected: Boolean(payload.ballDetected),
            },
          };
          if (payload.debugFrameUpdated) {
            details.progress_stats.debug_frame_url = `/api/analyses/${analysisId}/debug-frame?t=${Date.now()}`;
          }
          updateAnalysisProgress(analysisId, payload.progress, payload.message, details);
          onProgress?.(payload.progress, payload.message, details);
        }

        if (payload.type === 'result') {
          finalResult = payload.result;
        }

        if (payload.type === 'error') {
          workerError = payload.message || 'Erro no worker YOLO.';
        }
      };

      child.stdout.on('data', (chunk) => {
        stdoutBuffer += chunk.toString();
        const lines = stdoutBuffer.split(/\r?\n/);
        stdoutBuffer = lines.pop() || '';
        lines.forEach(handleLine);
      });

      child.stderr.on('data', (chunk) => {
        stderrBuffer += chunk.toString();
      });

      child.on('error', (err) => {
        if (err && (err.code === 'ENOENT' || err.code === 'EACCES')) {
          reject(
            new Error(
              `Não foi possível iniciar o Python (${python.command}). ${WINDOWS_STORE_ALIAS_HINT}`
            )
          );
          return;
        }
        reject(new Error(`Não foi possível iniciar o YOLO: ${err.message}`));
      });

      child.on('close', (code) => {
        if (stdoutBuffer) handleLine(stdoutBuffer);

        if (code !== 0 || workerError) {
          const combined = `${stdoutBuffer}\n${stderrBuffer}`;
          const looksLikeStoreAlias = /Microsoft Store|Aliases de execução|App execution aliases|Python (?:was not found|não foi encontrado)/i.test(
            combined
          );

          let detail;
          if (looksLikeStoreAlias) {
            detail = WINDOWS_STORE_ALIAS_HINT;
          } else {
            detail = workerError || stderrBuffer.trim() || `Processo YOLO finalizou com código ${code}.`;
          }
          reject(new Error(detail));
          return;
        }

        if (!finalResult) {
          reject(new Error('YOLO não retornou resultado final.'));
          return;
        }

        resolve(finalResult);
      });
    });
  }
}

module.exports = YoloAnalysisProvider;
