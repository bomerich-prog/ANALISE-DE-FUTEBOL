import { checkAuth, logout, api } from './api.js';
import { navigateTo } from './utils.js';

const PAGE_META = {
  dashboard: { title: 'Dashboard', breadcrumb: 'Visão geral da plataforma' },
  converter: { title: 'Converter Vídeos', breadcrumb: 'Módulo de ingestão de vídeo' },
  history: { title: 'Meu Histórico', breadcrumb: 'Biblioteca de vídeos convertidos' },
  analyses: { title: 'Análises', breadcrumb: 'Histórico de execuções de análise' },
  settings: { title: 'Configurações', breadcrumb: 'Conta e preferências' },
};

let currentUser = null;

const topbarEmail = document.getElementById('topbarEmail');
const userAvatar = document.getElementById('userAvatar');
const pageTitle = document.getElementById('pageTitle');
const pageBreadcrumb = document.getElementById('pageBreadcrumb');
const indicatorVideos = document.getElementById('indicatorVideos');
const indicatorAnalyses = document.getElementById('indicatorAnalyses');
const logoutBtn = document.getElementById('logoutBtn');
const navItems = document.querySelectorAll('.nav-item[data-view]');
const views = document.querySelectorAll('.view');

function setUserUI(user) {
  const initial = (user.email[0] || 'A').toUpperCase();
  topbarEmail.textContent = user.email;
  userAvatar.textContent = initial;
  const settingsAvatar = document.getElementById('settingsAvatar');
  if (settingsAvatar) settingsAvatar.textContent = initial;
}

async function refreshIndicators() {
  try {
    const data = await api('/api/dashboard/stats');
    indicatorVideos.textContent = String(data.totalVideos);
    if (indicatorAnalyses) indicatorAnalyses.textContent = String(data.totalAnalyses ?? 0);
    window.dispatchEvent(new CustomEvent('indicators:updated', { detail: data }));
  } catch {
    indicatorVideos.textContent = '—';
    if (indicatorAnalyses) indicatorAnalyses.textContent = '—';
  }
}

function setPageMeta(view) {
  const meta = PAGE_META[view] || PAGE_META.dashboard;
  pageTitle.textContent = meta.title;
  pageBreadcrumb.textContent = meta.breadcrumb;
  document.title = `${meta.title} — PitchView`;
}

async function init() {
  currentUser = await checkAuth();
  if (!currentUser) {
    window.location.href = '/login.html';
    return;
  }
  setUserUI(currentUser);
  await refreshIndicators();
}

navItems.forEach((btn) => {
  btn.addEventListener('click', () => {
    const view = btn.dataset.view;
    navItems.forEach((b) => b.classList.toggle('active', b === btn));
    views.forEach((v) => v.classList.toggle('active', v.id === `view-${view}`));
    setPageMeta(view);

    if (view === 'dashboard') window.dispatchEvent(new CustomEvent('dashboard:load'));
    if (view === 'history') window.dispatchEvent(new CustomEvent('history:load'));
    if (view === 'analyses') window.dispatchEvent(new CustomEvent('analyses:load'));
    if (view === 'settings') window.dispatchEvent(new CustomEvent('settings:load'));
  });
});

document.querySelectorAll('[data-goto]').forEach((btn) => {
  btn.addEventListener('click', () => navigateTo(btn.dataset.goto));
});

logoutBtn.addEventListener('click', logout);

window.addEventListener('dashboard:refresh', refreshIndicators);

document.querySelector('.btn-upload-trigger')?.addEventListener('click', (e) => {
  e.stopPropagation();
  document.getElementById('fileInput')?.click();
});

init();

export { currentUser, refreshIndicators };
