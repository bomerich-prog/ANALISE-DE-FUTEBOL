const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

const dataDir = path.join(__dirname, '..', 'data');
const dbPath = path.join(dataDir, 'app.db');

if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

const uploadsDir = path.join(dataDir, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

const db = new Database(dbPath);

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS conversions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    video_name TEXT NOT NULL,
    original_format TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    converted_at TEXT DEFAULT (datetime('now')),
    file_path TEXT NOT NULL,
    pipeline_status TEXT DEFAULT 'converted',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    video_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    result_json TEXT DEFAULT '{}',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (video_id) REFERENCES conversions(id) ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS idx_conversions_user ON conversions(user_id);
  CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id);
  CREATE INDEX IF NOT EXISTS idx_analyses_video ON analyses(video_id);
`);

try {
  db.exec(`ALTER TABLE conversions ADD COLUMN pipeline_status TEXT DEFAULT 'converted'`);
} catch {
  /* coluna já existe */
}

module.exports = { db, uploadsDir, dataDir };
