/** Espelha server/constants/pipelineStatus.js — manter sincronizado. */
export const PIPELINE_STATUS = {
  SENT: 'sent',
  CONVERTING: 'converting',
  CONVERTED: 'converted',
  ANALYZING: 'analyzing',
  ANALYSIS_COMPLETE: 'analysis_complete',
  ERROR: 'error',
};

export const ANALYSIS_STATUS = {
  PENDING: 'pending',
  ANALYZING: 'analyzing',
  COMPLETED: 'completed',
  ERROR: 'error',
};

export const PIPELINE_LABELS = {
  [PIPELINE_STATUS.SENT]: 'Enviado',
  [PIPELINE_STATUS.CONVERTING]: 'Convertendo',
  [PIPELINE_STATUS.CONVERTED]: 'Convertido',
  [PIPELINE_STATUS.ANALYZING]: 'Analisando',
  [PIPELINE_STATUS.ANALYSIS_COMPLETE]: 'Análise concluída',
  [PIPELINE_STATUS.ERROR]: 'Erro',
};

export const ANALYSIS_LABELS = {
  [ANALYSIS_STATUS.PENDING]: 'Pendente',
  [ANALYSIS_STATUS.ANALYZING]: 'Analisando',
  [ANALYSIS_STATUS.COMPLETED]: 'Concluída',
  [ANALYSIS_STATUS.ERROR]: 'Erro',
};

export const PIPELINE_STEPS = [
  PIPELINE_STATUS.SENT,
  PIPELINE_STATUS.CONVERTING,
  PIPELINE_STATUS.CONVERTED,
  PIPELINE_STATUS.ANALYZING,
  PIPELINE_STATUS.ANALYSIS_COMPLETE,
];

export function pipelinePillClass(status) {
  const map = {
    [PIPELINE_STATUS.SENT]: 'sent',
    [PIPELINE_STATUS.CONVERTING]: 'converting',
    [PIPELINE_STATUS.CONVERTED]: 'converted',
    [PIPELINE_STATUS.ANALYZING]: 'analyzing',
    [PIPELINE_STATUS.ANALYSIS_COMPLETE]: 'analysis-complete',
    [PIPELINE_STATUS.ERROR]: 'error',
  };
  return map[status] || 'sent';
}

export function getPipelineLabel(status) {
  return PIPELINE_LABELS[status] || status;
}

export function getAnalysisLabel(status) {
  return ANALYSIS_LABELS[status] || status;
}
