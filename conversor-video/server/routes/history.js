const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { randomUUID } = require('crypto');
const { db, uploadsDir } = require('../db');
const { requireAuth } = require('../middleware');
const { PIPELINE_STATUS } = require('../constants/pipelineStatus');

const router = express.Router();

const storage = multer.diskStorage({
  destination(req, file, cb) {
    const userDir = path.join(uploadsDir, String(req.session.userId));
    if (!fs.existsSync(userDir)) {
      fs.mkdirSync(userDir, { recursive: true });
    }
    cb(null, userDir);
  },
  filename(req, file, cb) {
    cb(null, `${randomUUID()}.mp4`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024 },
  fileFilter(req, file, cb) {
    if (file.mimetype === 'video/mp4' || file.originalname.toLowerCase().endsWith('.mp4')) {
      cb(null, true);
    } else {
      cb(new Error('Apenas arquivos MP4 são aceitos para o histórico.'));
    }
  },
});

router.get('/', requireAuth, (req, res) => {
  const rows = db
    .prepare(
      `SELECT c.id, c.video_name, c.original_format, c.file_size, c.converted_at,
              c.pipeline_status,
              (SELECT a.status FROM analyses a
               WHERE a.video_id = c.id AND a.user_id = c.user_id
               ORDER BY a.created_at DESC LIMIT 1) AS latest_analysis_status
       FROM conversions c
       WHERE c.user_id = ?
       ORDER BY c.converted_at DESC`
    )
    .all(req.session.userId);

  res.json({ history: rows });
});

router.post('/', requireAuth, upload.single('video'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Arquivo MP4 é obrigatório.' });
    }

    const { videoName, originalFormat } = req.body;

    if (!videoName?.trim()) {
      fs.unlinkSync(req.file.path);
      return res.status(400).json({ error: 'Nome do vídeo é obrigatório.' });
    }

    const result = db
      .prepare(
        `INSERT INTO conversions (user_id, video_name, original_format, file_size, file_path, pipeline_status)
         VALUES (?, ?, ?, ?, ?, ?)`
      )
      .run(
        req.session.userId,
        videoName.trim(),
        originalFormat?.trim() || 'desconhecido',
        req.file.size,
        req.file.path,
        PIPELINE_STATUS.CONVERTED
      );

    res.status(201).json({
      message: 'Vídeo salvo no histórico.',
      item: {
        id: result.lastInsertRowid,
        video_name: videoName.trim(),
        original_format: originalFormat?.trim() || 'desconhecido',
        file_size: req.file.size,
        converted_at: new Date().toISOString(),
        pipeline_status: PIPELINE_STATUS.CONVERTED,
      },
    });
  } catch (err) {
    if (req.file?.path && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    console.error('Erro ao salvar histórico:', err);
    res.status(500).json({ error: 'Erro ao salvar vídeo no histórico.' });
  }
});

router.patch('/:id/pipeline-status', requireAuth, (req, res) => {
  const { status } = req.body;
  const allowed = Object.values(PIPELINE_STATUS);
  if (!allowed.includes(status)) {
    return res.status(400).json({ error: 'Status inválido.' });
  }

  const result = db
    .prepare('UPDATE conversions SET pipeline_status = ? WHERE id = ? AND user_id = ?')
    .run(status, req.params.id, req.session.userId);

  if (result.changes === 0) {
    return res.status(404).json({ error: 'Vídeo não encontrado.' });
  }

  res.json({ message: 'Status atualizado.', pipeline_status: status });
});

router.get('/:id/download', requireAuth, (req, res) => {
  const item = db
    .prepare('SELECT * FROM conversions WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.session.userId);

  if (!item) {
    return res.status(404).json({ error: 'Vídeo não encontrado.' });
  }

  if (!fs.existsSync(item.file_path)) {
    return res.status(404).json({ error: 'Arquivo não encontrado no servidor.' });
  }

  const safeName = item.video_name.replace(/[^\w\s.-]/g, '_');
  res.download(item.file_path, `${safeName}.mp4`);
});

router.delete('/:id', requireAuth, (req, res) => {
  const item = db
    .prepare('SELECT * FROM conversions WHERE id = ? AND user_id = ?')
    .get(req.params.id, req.session.userId);

  if (!item) {
    return res.status(404).json({ error: 'Vídeo não encontrado.' });
  }

  if (fs.existsSync(item.file_path)) {
    fs.unlinkSync(item.file_path);
  }

  db.prepare('DELETE FROM conversions WHERE id = ?').run(item.id);
  res.json({ message: 'Vídeo removido do histórico.' });
});

module.exports = router;
