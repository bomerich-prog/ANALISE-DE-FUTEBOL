const express = require('express');
const { db } = require('../db');
const { requireAuth } = require('../middleware');
const { ANALYSIS_STATUS } = require('../constants/pipelineStatus');

const router = express.Router();

router.get('/stats', requireAuth, (req, res) => {
  const userId = req.session.userId;

  const totalVideos = db
    .prepare('SELECT COUNT(*) AS count FROM conversions WHERE user_id = ?')
    .get(userId).count;

  const totalStorage = db
    .prepare('SELECT COALESCE(SUM(file_size), 0) AS total FROM conversions WHERE user_id = ?')
    .get(userId).total;

  const totalAnalyses = db
    .prepare('SELECT COUNT(*) AS count FROM analyses WHERE user_id = ?')
    .get(userId).count;

  const completedAnalyses = db
    .prepare(
      `SELECT COUNT(*) AS count FROM analyses WHERE user_id = ? AND status = ?`
    )
    .get(userId, ANALYSIS_STATUS.COMPLETED).count;

  const lastAnalysis = db
    .prepare(
      `SELECT created_at FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT 1`
    )
    .get(userId);

  const successRate =
    totalAnalyses > 0 ? Math.round((completedAnalyses / totalAnalyses) * 100) : 0;

  const recent = db
    .prepare(
      `SELECT id, video_name, original_format, file_size, converted_at, pipeline_status
       FROM conversions WHERE user_id = ?
       ORDER BY converted_at DESC LIMIT 8`
    )
    .all(userId);

  const recentAnalyses = db
    .prepare(
      `SELECT a.id, a.status, a.created_at, c.video_name
       FROM analyses a
       JOIN conversions c ON c.id = a.video_id
       WHERE a.user_id = ?
       ORDER BY a.created_at DESC LIMIT 8`
    )
    .all(userId);

  const formatBreakdown = db
    .prepare(
      `SELECT original_format, COUNT(*) AS count
       FROM conversions WHERE user_id = ?
       GROUP BY original_format ORDER BY count DESC LIMIT 6`
    )
    .all(userId);

  const last7Days = db
    .prepare(
      `SELECT date(converted_at) AS day, COUNT(*) AS count
       FROM conversions
       WHERE user_id = ? AND converted_at >= datetime('now', '-7 days')
       GROUP BY date(converted_at)
       ORDER BY day ASC`
    )
    .all(userId);

  res.json({
    totalVideos,
    totalConversions: totalVideos,
    totalStorage,
    totalAnalyses,
    completedAnalyses,
    successRate,
    lastAnalysisAt: lastAnalysis?.created_at || null,
    recent,
    recentAnalyses,
    formatBreakdown,
    activity: last7Days,
  });
});

module.exports = router;
