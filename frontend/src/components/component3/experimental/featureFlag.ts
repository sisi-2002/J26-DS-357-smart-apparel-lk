/** Build-time opt-in; unset and all values except "true" keep the preview off. */
export function experimentalPreviewEnabled(): boolean {
  return import.meta.env.VITE_C3_EXPERIMENTAL_MULTIHORIZON === 'true';
}
