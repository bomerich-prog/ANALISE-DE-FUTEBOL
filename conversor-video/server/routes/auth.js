const express = require('express');
const bcrypt = require('bcrypt');
const { db } = require('../db');

const router = express.Router();
const SALT_ROUNDS = 10;

function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

router.post('/register', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email?.trim() || !password) {
      return res.status(400).json({ error: 'Email e senha são obrigatórios.' });
    }

    const normalizedEmail = email.trim().toLowerCase();

    if (!isValidEmail(normalizedEmail)) {
      return res.status(400).json({ error: 'Email inválido.' });
    }

    if (password.length < 6) {
      return res.status(400).json({ error: 'A senha deve ter pelo menos 6 caracteres.' });
    }

    const existing = db.prepare('SELECT id FROM users WHERE email = ?').get(normalizedEmail);
    if (existing) {
      return res.status(409).json({ error: 'Este email já está cadastrado.' });
    }

    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
    const result = db
      .prepare('INSERT INTO users (email, password_hash) VALUES (?, ?)')
      .run(normalizedEmail, passwordHash);

    req.session.userId = result.lastInsertRowid;
    req.session.email = normalizedEmail;

    res.status(201).json({
      message: 'Cadastro realizado com sucesso.',
      user: { id: result.lastInsertRowid, email: normalizedEmail },
    });
  } catch (err) {
    console.error('Erro no cadastro:', err);
    res.status(500).json({ error: 'Erro interno ao cadastrar usuário.' });
  }
});

router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email?.trim() || !password) {
      return res.status(400).json({ error: 'Email e senha são obrigatórios.' });
    }

    const normalizedEmail = email.trim().toLowerCase();
    const user = db.prepare('SELECT * FROM users WHERE email = ?').get(normalizedEmail);

    if (!user) {
      return res.status(401).json({ error: 'Email ou senha incorretos.' });
    }

    const valid = await bcrypt.compare(password, user.password_hash);
    if (!valid) {
      return res.status(401).json({ error: 'Email ou senha incorretos.' });
    }

    req.session.userId = user.id;
    req.session.email = user.email;

    res.json({
      message: 'Login realizado com sucesso.',
      user: { id: user.id, email: user.email },
    });
  } catch (err) {
    console.error('Erro no login:', err);
    res.status(500).json({ error: 'Erro interno ao fazer login.' });
  }
});

router.post('/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      return res.status(500).json({ error: 'Erro ao encerrar sessão.' });
    }
    res.clearCookie('connect.sid');
    res.json({ message: 'Logout realizado com sucesso.' });
  });
});

router.get('/me', (req, res) => {
  if (!req.session?.userId) {
    return res.status(401).json({ error: 'Não autenticado.' });
  }
  const user = db.prepare('SELECT id, email, created_at FROM users WHERE id = ?').get(req.session.userId);
  res.json({
    user: {
      id: user.id,
      email: user.email,
      created_at: user.created_at,
    },
  });
});

module.exports = router;
