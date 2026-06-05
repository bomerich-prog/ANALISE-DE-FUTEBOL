const express = require('express');
const fs = require('fs');
const path = require('path');
const analysisService = require('../services/analysis/AnalysisService');
const { requireAuth } = require('../middleware');

const router = express.Router();

router.get('/', requireAuth, (req, res) => {
  const analyses = analysisService.listAnalyses(req.session.userId);
  res.json({ analyses });
});

router.post('/start', requireAuth, async (req, res) => {
  try {
    const { videoId } = req.body;
    if (!videoId) {
      return res.status(400).json({ error: 'videoId é obrigatório.' });
    }

    const result = await analysisService.startAnalysis(req.session.userId, Number(videoId));
    res.status(201).json(result);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

router.get('/video/:videoId/latest', requireAuth, (req, res) => {
  const video = analysisService.getVideoForAnalysis(req.session.userId, Number(req.params.videoId));
  if (!video) {
    return res.status(404).json({ error: 'Vídeo não encontrado.' });
  }

  const analysis = analysisService.getLatestAnalysisForVideo(
    req.session.userId,
    Number(req.params.videoId)
  );

  res.json({ video, analysis });
});

router.get('/:id/video', requireAuth, (req, res) => {
  const analysisId = Number(req.params.id);
  const videoPath = analysisService.getAnalysisVideoPath(req.session.userId, analysisId);

  if (!videoPath || !fs.existsSync(videoPath)) {
    return res.status(404).json({ error: 'Vídeo processado não encontrado.' });
  }

  const stat = fs.statSync(videoPath);
  const fileSize = stat.size;
  const range = req.headers.range;
  const safeName = path.basename(videoPath);

  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Content-Disposition', `inline; filename="${safeName}"`);

  if (!range) {
    res.setHeader('Content-Length', fileSize);
    fs.createReadStream(videoPath).pipe(res);
    return;
  }

  const [startRaw, endRaw] = range.replace(/bytes=/, '').split('-');
  const start = Number.parseInt(startRaw, 10);
  const end = endRaw ? Number.parseInt(endRaw, 10) : fileSize - 1;

  if (Number.isNaN(start) || Number.isNaN(end) || start >= fileSize || end >= fileSize) {
    res.status(416).setHeader('Content-Range', `bytes */${fileSize}`);
    res.end();
    return;
  }

  res.status(206);
  res.setHeader('Content-Range', `bytes ${start}-${end}/${fileSize}`);
  res.setHeader('Content-Length', end - start + 1);
  fs.createReadStream(videoPath, { start, end }).pipe(res);
});

router.get('/:id', requireAuth, (req, res) => {
  const analysis = analysisService.getAnalysis(req.session.userId, Number(req.params.id));
  if (!analysis) {
    return res.status(404).json({ error: 'Análise não encontrada.' });
  }
  res.json({ analysis });
});

module.exports = router;
