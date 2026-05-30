const express = require('express');
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

router.get('/:id', requireAuth, (req, res) => {
  const analysis = analysisService.getAnalysis(req.session.userId, Number(req.params.id));
  if (!analysis) {
    return res.status(404).json({ error: 'Análise não encontrada.' });
  }
  res.json({ analysis });
});

module.exports = router;
