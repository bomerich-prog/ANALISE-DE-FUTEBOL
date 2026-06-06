const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const IS_WINDOWS = process.platform === 'win32';

/**
 * No Windows, executar diretamente `python` ou `python3` pode acionar o
 * "App Execution Alias" do Microsoft Store, que NÃO instala o Python e ainda
 * imprime a mensagem "Python não foi encontrado..." quando recebe argumentos.
 *
 * Esses stubs ficam em:
 *   %LocalAppData%\Microsoft\WindowsApps\python.exe
 *   %LocalAppData%\Microsoft\WindowsApps\python3.exe
 *
 * Detectamos pelo caminho contendo "WindowsApps" e descartamos a opção.
 */
function isWindowsStoreAliasPath(executablePath) {
  if (!executablePath) return false;
  return /\\WindowsApps\\/i.test(executablePath);
}

/**
 * Resolve o caminho real de um executável usando o comando do SO.
 * Retorna a primeira linha não vazia ou null.
 */
function resolveExecutablePath(command) {
  const finder = IS_WINDOWS ? 'where' : 'which';
  const result = spawnSync(finder, [command], { encoding: 'utf8' });
  if (result.status !== 0 || !result.stdout) return null;
  const firstLine = result.stdout.split(/\r?\n/).find((line) => line.trim());
  return firstLine ? firstLine.trim() : null;
}

/**
 * Lista de candidatos por plataforma, em ordem de preferência.
 * Cada item é { command, args } para suportar entradas como `py -3`.
 */
function getCandidates() {
  if (IS_WINDOWS) {
    return [
      { command: 'py', args: ['-3'] },
      { command: 'py', args: [] },
      { command: 'python', args: [] },
      { command: 'python3', args: [] },
    ];
  }
  return [
    { command: 'python3', args: [] },
    { command: 'python', args: [] },
  ];
}

/**
 * Faz o parse de uma string como "py -3" ou "C:\\Python\\python.exe".
 */
function parseUserProvidedBin(value) {
  if (!value) return null;
  const trimmed = String(value).trim();
  if (!trimmed) return null;
  const parts = trimmed.match(/(?:[^\s"]+|"[^"]*")+/g) || [trimmed];
  const [command, ...args] = parts.map((part) => part.replace(/^"|"$/g, ''));
  return { command, args };
}

/**
 * Verifica se um candidato responde a `--version` com sucesso.
 * Também garante que NÃO seja o stub do Microsoft Store no Windows.
 */
function probeCandidate(candidate) {
  if (IS_WINDOWS && !path.isAbsolute(candidate.command)) {
    const resolved = resolveExecutablePath(candidate.command);
    if (resolved && isWindowsStoreAliasPath(resolved)) {
      return { ok: false, reason: 'windows-store-alias', resolvedPath: resolved };
    }
  }

  if (path.isAbsolute(candidate.command) && !fs.existsSync(candidate.command)) {
    return { ok: false, reason: 'not-found' };
  }

  const probe = spawnSync(candidate.command, [...candidate.args, '--version'], {
    encoding: 'utf8',
    windowsHide: true,
  });

  if (probe.error) {
    return { ok: false, reason: probe.error.code || 'spawn-error', error: probe.error };
  }
  if (probe.status !== 0) {
    return { ok: false, reason: `exit-${probe.status}`, stderr: probe.stderr };
  }

  const versionOutput = `${probe.stdout || ''}${probe.stderr || ''}`.trim();
  return { ok: true, version: versionOutput };
}

let cached = null;

/**
 * Resolve o interpretador Python a ser usado pelo worker YOLO.
 *
 * Ordem de prioridade:
 *   1. `options.pythonBin` (passado pelo provider)
 *   2. `process.env.YOLO_PYTHON_BIN`
 *   3. Lista de candidatos por plataforma (com filtro do stub do Store)
 *
 * Retorna sempre `{ command, args, source }`. Quando nada funciona, lança um
 * Error com mensagem explicativa em português.
 */
function resolvePython(options = {}) {
  const attempts = [];
  const userProvided =
    parseUserProvidedBin(options.pythonBin) ||
    parseUserProvidedBin(process.env.YOLO_PYTHON_BIN);

  if (userProvided) {
    const probe = probeCandidate(userProvided);
    attempts.push({ candidate: userProvided, probe, source: 'configured' });
    if (probe.ok) {
      return { ...userProvided, source: 'configured', version: probe.version, attempts };
    }

    const label = [userProvided.command, ...userProvided.args].join(' ');
    const hint = IS_WINDOWS
      ? 'Verifique se o caminho está correto e se não aponta para o atalho do Microsoft Store em %LocalAppData%\\Microsoft\\WindowsApps. Tente algo como YOLO_PYTHON_BIN="py -3" ou YOLO_PYTHON_BIN="C:\\\\Caminho\\\\para\\\\python.exe".'
      : 'Verifique se o caminho está correto e se o arquivo tem permissão de execução.';
    const reason =
      probe.reason === 'windows-store-alias'
        ? `aponta para o atalho do Microsoft Store em ${probe.resolvedPath}`
        : probe.reason;
    const err = new Error(
      `YOLO_PYTHON_BIN aponta para "${label}", mas não foi possível executá-lo (${reason}). ${hint}`
    );
    err.code = 'PYTHON_NOT_FOUND';
    err.attempts = attempts;
    throw err;
  }

  for (const candidate of getCandidates()) {
    const probe = probeCandidate(candidate);
    attempts.push({ candidate, probe, source: 'auto' });
    if (probe.ok) {
      return { ...candidate, source: 'auto', version: probe.version, attempts };
    }
  }

  const detail = attempts
    .map(({ candidate, probe }) => {
      const label = [candidate.command, ...candidate.args].join(' ');
      if (probe.reason === 'windows-store-alias') {
        return `${label} → atalho do Microsoft Store em ${probe.resolvedPath}`;
      }
      return `${label} → ${probe.reason}`;
    })
    .join('; ');

  const hint = IS_WINDOWS
    ? 'Instale Python (https://www.python.org/downloads/) ou desabilite os "Aliases de execução do aplicativo" para python.exe/python3.exe em Configurações > Aplicativos. Depois reinicie o terminal ou defina YOLO_PYTHON_BIN apontando para o python.exe correto.'
    : 'Instale Python 3 ou defina YOLO_PYTHON_BIN com o caminho do interpretador.';

  const err = new Error(`Não foi possível localizar um interpretador Python para a análise YOLO. ${hint} Tentativas: ${detail}.`);
  err.code = 'PYTHON_NOT_FOUND';
  err.attempts = attempts;
  throw err;
}

function resolvePythonCached(options = {}) {
  if (cached) return cached;
  cached = resolvePython(options);
  return cached;
}

function resetCacheForTests() {
  cached = null;
}

module.exports = {
  resolvePython,
  resolvePythonCached,
  isWindowsStoreAliasPath,
  parseUserProvidedBin,
  resetCacheForTests,
};
