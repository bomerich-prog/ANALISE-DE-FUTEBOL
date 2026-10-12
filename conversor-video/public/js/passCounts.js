/** Group backend pass events by the team already stored on each event. */

export function countPassesByTeam(passes) {
  const events = Array.isArray(passes?.events) ? passes.events : [];
  let team0 = 0;
  let team1 = 0;
  for (const event of events) {
    if (event?.team === 'Team 0') team0 += 1;
    else if (event?.team === 'Team 1') team1 += 1;
  }
  return { team0, team1 };
}
