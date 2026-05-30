import { api } from './api.js';

const form = document.getElementById('authForm');
const errorBox = document.getElementById('errorBox');
const submitBtn = document.getElementById('submitBtn');

const isRegister = document.body.dataset.page === 'register';

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
}

function hideError() {
  errorBox.classList.add('hidden');
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideError();

  const email = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;

  submitBtn.disabled = true;
  submitBtn.textContent = isRegister ? 'Cadastrando...' : 'Entrando...';

  try {
    const endpoint = isRegister ? '/api/auth/register' : '/api/auth/login';
    await api(endpoint, {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    window.location.href = '/app.html';
  } catch (err) {
    showError(err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = isRegister ? 'Criar conta' : 'Entrar';
  }
});
