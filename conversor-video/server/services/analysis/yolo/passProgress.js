/** Copy the pass totals the detector already produced into a progress payload. */

function passProgressFromPayload(payload) {
  const passes = payload && payload.passes;
  if (!passes || typeof passes !== 'object') return null;
  const turnovers = Number(passes.turnovers);
  return {
    total_passes: Number(passes.total_passes) || 0,
    events: Array.isArray(passes.events) ? passes.events : [],
    turnovers: Number.isFinite(turnovers) && turnovers > 0 ? Math.round(turnovers) : 0,
  };
}

module.exports = { passProgressFromPayload };
