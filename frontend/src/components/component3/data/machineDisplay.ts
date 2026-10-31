/** Illustrative presentation metadata only; never part of an inference request. */
export function getMachineDisplay(machineId: number) {
  if (!Number.isInteger(machineId) || machineId < 1 || machineId > 100) {
    throw new RangeError('Machine ID must be an integer from 1 to 100.');
  }
  // Ten illustrative lines, with ten consecutive benchmark IDs per line.
  const lineNumber = Math.floor((machineId - 1) / 10) + 1;
  return {
    name: `Machine ${String(machineId).padStart(3, '0')}`,
    line: `Line ${String(lineNumber).padStart(2, '0')}`,
  };
}
