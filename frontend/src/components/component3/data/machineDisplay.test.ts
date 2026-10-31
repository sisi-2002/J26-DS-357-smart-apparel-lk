import { describe, expect, it } from 'vitest';
import { MACHINE_IDS } from './demoOptions';
import { getMachineDisplay } from './machineDisplay';

describe('illustrative machine display metadata', () => {
  it.each([
    [1, 'Machine 001', 'Line 01'], [10, 'Machine 010', 'Line 01'],
    [11, 'Machine 011', 'Line 02'], [22, 'Machine 022', 'Line 03'],
    [64, 'Machine 064', 'Line 07'], [71, 'Machine 071', 'Line 08'],
    [91, 'Machine 091', 'Line 10'], [100, 'Machine 100', 'Line 10'],
  ])('labels machine %i consistently', (id, name, line) => {
    expect(getMachineDisplay(Number(id))).toEqual({ name, line });
  });

  it('covers all 100 machines with unique names and ten stable, equally sized illustrative lines', () => {
    const displays = MACHINE_IDS.map(getMachineDisplay);
    expect(new Set(displays.map(machine => machine.name)).size).toBe(100);
    const lines = new Set(displays.map(machine => machine.line));
    expect(lines.size).toBe(10);
    for (const line of lines) expect(displays.filter(machine => machine.line === line)).toHaveLength(10);
    expect([...MACHINE_IDS].reverse().map(getMachineDisplay).reverse()).toEqual(displays);
    expect(MACHINE_IDS.map(getMachineDisplay)).toEqual(displays);
    expect(displays.some(machine => /demo/i.test(`${machine.name} ${machine.line}`))).toBe(false);
  });

  it.each([0, -1, 101, 1.5, NaN])('does not invent display metadata for invalid ID %s', id => {
    expect(() => getMachineDisplay(id)).toThrow(RangeError);
  });
});
