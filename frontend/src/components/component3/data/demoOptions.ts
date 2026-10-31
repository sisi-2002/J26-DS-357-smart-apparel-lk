// Public benchmark input metadata only. No predictions, source events, or models.
// FastAPI remains authoritative for machine existence and timestamp validity.
export const MACHINE_IDS = Array.from({ length: 100 }, (_, index) => index + 1);
export const PREDICTION_TIMESTAMPS = [
  '2015-09-02T06:00:00', '2015-09-09T06:00:00', '2015-09-16T06:00:00',
  '2015-09-23T06:00:00', '2015-09-30T06:00:00', '2015-10-07T06:00:00',
  '2015-10-14T06:00:00', '2015-10-21T06:00:00', '2015-10-28T06:00:00',
  '2015-11-04T06:00:00', '2015-11-11T06:00:00', '2015-11-18T06:00:00',
  '2015-11-25T06:00:00',
] as const;

/** Format naive benchmark strings without assigning a timezone or using Date. */
export function formatTimestamp(timestamp: string): string {
  const [date] = timestamp.split(/[T ]/);
  const [year, month, day] = date.split('-');
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${Number(day)} ${months[Number(month) - 1]} ${year} · 06:00`;
}
