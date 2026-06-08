#!/usr/bin/env node
/**
 * Verifica o resolvedor de Python contra o cenário do alias do Microsoft Store.
 *
 * Estratégia:
 *  - Cria um diretório temporário com um shim `python3` que imprime a mesma
 *    mensagem do stub do Windows e sai com código != 0.
 *  - Cria também um shim `python` válido (delega para o python3 real do sistema).
 *  - Prepõe o diretório ao PATH e limpa o cache do resolver.
 *  - Verifica que o resolvedor escolhe `python` (e não o shim `python3`).
 *  - Em seguida, força YOLO_PYTHON_BIN=python3 e verifica que o erro retornado
 *    contém a dica útil em português.
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const resolverPath = require.resolve('../server/services/analysis/yolo/pythonResolver');
const realPython = spawnSync('which', ['python3']).stdout.toString().trim();
if (!realPython) {
  console.error('Não foi possível encontrar o python3 do sistema para o teste.');
  process.exit(1);
}

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'yolo-resolver-'));

const fakePython3 = path.join(tmpDir, 'python3');
fs.writeFileSync(
  fakePython3,
  '#!/usr/bin/env bash\n' +
    'echo "Python não foi encontrado; executar sem argumentos para instalar do Microsoft Store ou desabilitar este atalho em Configurações > Aplicativos > Configurações avançadas do aplicativo > Aliases de execução do aplicativo." 1>&2\n' +
    'exit 9009\n'
);
fs.chmodSync(fakePython3, 0o755);

const goodPython = path.join(tmpDir, 'python');
fs.writeFileSync(goodPython, `#!/usr/bin/env bash\nexec "${realPython}" "$@"\n`);
fs.chmodSync(goodPython, 0o755);

process.env.PATH = `${tmpDir}:${process.env.PATH}`;
delete process.env.YOLO_PYTHON_BIN;
delete require.cache[resolverPath];
const resolver = require(resolverPath);

let failures = 0;

function expect(label, condition, detail = '') {
  if (condition) {
    console.log(`  ok  ${label}`);
  } else {
    console.log(`  FAIL ${label}${detail ? ` — ${detail}` : ''}`);
    failures += 1;
  }
}

console.log('\n[caso 1] python3 falhando como stub; python válido no PATH');
resolver.resetCacheForTests();
const result1 = resolver.resolvePython();
expect('escolhe python ao invés do shim python3', result1.command === 'python', `recebeu ${result1.command}`);
expect('marca origem como auto', result1.source === 'auto');
expect('inclui versão detectada', /Python /.test(result1.version || ''));

console.log('\n[caso 2] usuário força YOLO_PYTHON_BIN=python3 (stub) — deve falhar com mensagem útil');
resolver.resetCacheForTests();
let err2;
try {
  resolver.resolvePython({ pythonBin: 'python3' });
} catch (e) {
  err2 = e;
}
expect('lança erro', err2 instanceof Error);
expect('código PYTHON_NOT_FOUND', err2 && err2.code === 'PYTHON_NOT_FOUND');
expect(
  'mensagem orienta o usuário (PT-BR)',
  err2 && /Python|YOLO_PYTHON_BIN/i.test(err2.message),
  err2 ? err2.message.slice(0, 160) : ''
);

console.log('\n[caso 3] parser de "py -3"');
const parsed = resolver.parseUserProvidedBin('py -3');
expect('command = py', parsed && parsed.command === 'py');
expect('args = ["-3"]', parsed && Array.isArray(parsed.args) && parsed.args[0] === '-3');

console.log('\n[caso 4] detecção de caminhos do WindowsApps');
expect(
  'caminho WindowsApps é reconhecido',
  resolver.isWindowsStoreAliasPath(
    'C\\:\\\\Users\\\\Emerson\\\\AppData\\\\Local\\\\Microsoft\\\\WindowsApps\\\\python3.exe'.replace(/\\\\/g, '\\')
  )
);
expect(
  'caminho normal não é reconhecido',
  !resolver.isWindowsStoreAliasPath('C:\\Python314\\python.exe')
);

fs.rmSync(tmpDir, { recursive: true, force: true });

if (failures > 0) {
  console.error(`\n${failures} verificação(ões) falharam.`);
  process.exit(1);
}
console.log('\nTodas as verificações passaram.');
