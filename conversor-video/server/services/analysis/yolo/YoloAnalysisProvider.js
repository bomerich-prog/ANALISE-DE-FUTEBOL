const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const { db, dataDir } = require('../../../db');
const { AnalysisProvider, updateAnalysisProgress } = require('../AnalysisProvider');

class YoloAnalysisProvider extends AnalysisProvider {
  constructor(options = {}) {
    super();
    this.pythonBin = options.pythonBin || process.env.YOLO_PYTHON_BIN || 'python3';
    this.model = options.model || process.env.YOLO_MODEL || 'yolov8n.pt';
    this.confidence = options.confidence || process.env.YOLO_CONFIDENCE || '0.25';
    this.imageSize = options.imageSize || process.env.YOLO_IMAGE_SIZE || '640';
    this.workerPath = options.workerPath || path.join(__dirname, 'run_yolo_analysis.py');
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

    const args = [
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

    return this.runWorker(args, analysisId, onProgress).then((result) => ({
      ...result,
      video_id: videoId,
      source_video_name: video.video_name,
    }));
  }

  runWorker(args, analysisId, onProgress) {
    return new Promise((resolve, reject) => {
      const child = spawn(this.pythonBin, args, {
        cwd: path.join(__dirname, '..', '..', '..', '..'),
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
        },
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
        reject(new Error(`Não foi possível iniciar o YOLO: ${err.message}`));
      });

      child.on('close', (code) => {
        if (stdoutBuffer) handleLine(stdoutBuffer);

        if (code !== 0 || workerError) {
          const detail = workerError || stderrBuffer.trim() || `Processo YOLO finalizou com código ${code}.`;
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
