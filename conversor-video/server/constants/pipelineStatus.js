/** Status do pipeline de processamento do vídeo (conversão + análise). */
const PIPELINE_STATUS = {
  SENT: 'sent',
  CONVERTING: 'converting',
  CONVERTED: 'converted',
  ANALYZING: 'analyzing',
  ANALYSIS_COMPLETE: 'analysis_complete',
  ERROR: 'error',
};

/** Status de uma execução de análise na tabela analyses. */
const ANALYSIS_STATUS = {
  PENDING: 'pending',
  ANALYZING: 'analyzing',
  COMPLETED: 'completed',
  ERROR: 'error',
};

const PIPELINE_LABELS = {
  [PIPELINE_STATUS.SENT]: 'Enviado',
  [PIPELINE_STATUS.CONVERTING]: 'Convertendo',
  [PIPELINE_STATUS.CONVERTED]: 'Convertido',
  [PIPELINE_STATUS.ANALYZING]: 'Analisando',
  [PIPELINE_STATUS.ANALYSIS_COMPLETE]: 'Análise concluída',
  [PIPELINE_STATUS.ERROR]: 'Erro',
};

const ANALYSIS_LABELS = {
  [ANALYSIS_STATUS.PENDING]: 'Pendente',
  [ANALYSIS_STATUS.ANALYZING]: 'Analisando',
  [ANALYSIS_STATUS.COMPLETED]: 'Concluída',
  [ANALYSIS_STATUS.ERROR]: 'Erro',
};

module.exports = {
  PIPELINE_STATUS,
  ANALYSIS_STATUS,
  PIPELINE_LABELS,
  ANALYSIS_LABELS,
};
