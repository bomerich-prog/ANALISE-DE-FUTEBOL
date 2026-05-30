import { checkAuth, api } from './api.js';
import { formatSize, formatDateShort } from './utils.js';

async function loadSettings() {
  const user = await checkAuth();
  if (!user) return;

  const initial = (user.email[0] || 'A').toUpperCase();
  document.getElementById('settingsAvatar').textContent = initial;
  document.getElementById('settingsEmail').textContent = user.email;
  document.getElementById('settingsSince').textContent = user.created_at
    ? formatDateShort(user.created_at)
    : '—';

  try {
    const data = await api('/api/dashboard/stats');
    document.getElementById('settingsVideos').textContent = String(data.totalVideos);
    document.getElementById('settingsStorage').textContent = formatSize(data.totalStorage);
  } catch {
    document.getElementById('settingsVideos').textContent = '—';
    document.getElementById('settingsStorage').textContent = '—';
  }
}

window.addEventListener('settings:load', loadSettings);
window.addEventListener('indicators:updated', (e) => {
  if (document.getElementById('view-settings')?.classList.contains('active')) {
    document.getElementById('settingsVideos').textContent = String(e.detail.totalVideos);
    document.getElementById('settingsStorage').textContent = formatSize(e.detail.totalStorage);
  }
});
