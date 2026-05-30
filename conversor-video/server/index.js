const express = require('express');
const session = require('express-session');
const path = require('path');

const authRoutes = require('./routes/auth');
const historyRoutes = require('./routes/history');
const dashboardRoutes = require('./routes/dashboard');
const analysesRoutes = require('./routes/analyses');

const app = express();
const PORT = process.env.PORT || 3000;
const publicDir = path.join(__dirname, '..', 'public');

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(
  session({
    secret: process.env.SESSION_SECRET || 'conversor-video-secret-change-in-production',
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      maxAge: 7 * 24 * 60 * 60 * 1000,
      sameSite: 'lax',
    },
  })
);

app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  next();
});

app.use('/api/auth', authRoutes);
app.use('/api/history', historyRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/analyses', analysesRoutes);

app.use(express.static(publicDir));

app.get('/', (req, res) => {
  if (req.session?.userId) {
    return res.redirect('/app.html');
  }
  res.redirect('/login.html');
});

app.use((err, req, res, next) => {
  if (err?.code === 'LIMIT_FILE_SIZE') {
    return res.status(400).json({ error: 'Arquivo muito grande (máximo 500 MB).' });
  }
  if (err) {
    console.error(err);
    return res.status(500).json({ error: err.message || 'Erro interno.' });
  }
  next();
});

app.listen(PORT, () => {
  console.log(`Servidor rodando em http://localhost:${PORT}`);
});
