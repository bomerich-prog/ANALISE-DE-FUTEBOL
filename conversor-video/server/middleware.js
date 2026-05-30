function requireAuth(req, res, next) {
  if (!req.session?.userId) {
    return res.status(401).json({ error: 'Não autenticado. Faça login novamente.' });
  }
  next();
}

module.exports = { requireAuth };
