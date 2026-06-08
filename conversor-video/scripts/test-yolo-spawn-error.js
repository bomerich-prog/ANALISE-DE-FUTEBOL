#!/usr/bin/env node
/**
 * Verifica o caminho de erro do YoloAnalysisProvider quando o Python
 * disponível é, na prática, o stub do Microsoft Store.
 */
const fs = require('fs');
const os = require('os');
const path = require('path');

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'yolo-spawn-'));
const fakePython = path.join(tmpDir, 'python_stub');
fs.writeFileSync(
  fakePython,
  '#!/usr/bin/env bash\n' +
    'echo "Python não foi encontrado; executar sem argumentos para instalar do Microsoft Store ou desabilitar este atalho em Configurações > Aplicativos > Configurações avançadas do aplicativo > Aliases de execução do aplicativo." 1>&2\n' +
    'exit 9009\n'
);
fs.chmodSync(fakePython, 0o755);

process.env.YOLO_PYTHON_BIN = fakePython;

const ProviderPath = require.resolve('../server/services/analysis/yolo/YoloAnalysisProvider');
const ResolverPath = require.resolve('../server/services/analysis/yolo/pythonResolver');

delete require.cache[ResolverPath];
delete require.cache[ProviderPath];

const dbModulePath = require.resolve('../server/db');
require.cache[dbModulePath] = {
  id: dbModulePath,
  filename: dbModulePath,
  loaded: true,
  exports: { db: {}, dataDir: tmpDir },
};

const analysisProviderPath = require.resolve('../server/services/analysis/AnalysisProvider');
require.cache[analysisProviderPath] = {
  id: analysisProviderPath,
  filename: analysisProviderPath,
  loaded: true,
  exports: {
    AnalysisProvider: class {},
    updateAnalysisProgress() {},
  },
};

const YoloAnalysisProvider = require(ProviderPath);
const provider = new YoloAnalysisProvider({ pythonBin: fakePython });

(async () => {
  let failed = false;
  try {
    await provider.runWorker(['fake_script.py'], 1, () => {});
    console.error('FAIL: expected runWorker to reject, but it resolved.');
    failed = true;
  } catch (err) {
    console.log('Erro recebido pelo provider:\n  ', err.message);
    const ok = /Microsoft Store|Aliases de execução|YOLO_PYTHON_BIN/i.test(err.message);
    if (!ok) {
      console.error('FAIL: mensagem não contém orientação esperada.');
      failed = true;
    } else {
      console.log('OK: mensagem traz orientação amigável.');
    }
  }
  fs.rmSync(tmpDir, { recursive: true, force: true });
  process.exit(failed ? 1 : 0);
})();
