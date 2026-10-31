import { apiFetch } from '../../../shared/api/client';
import type { Explanation, Inspection, InspectionListResponse } from '../types/component1';

export async function submitInspection(
  file: File,
  confidenceThreshold: number,
): Promise<Inspection> {
  const form = new FormData();
  form.append('file', file);

  const url = `/api/v1/component1/inspect?confidence_threshold=${confidenceThreshold}`;
  return apiFetch<Inspection>(url, { method: 'POST', body: form });
}

export async function getInspection(id: string): Promise<Inspection> {
  return apiFetch<Inspection>(`/api/v1/component1/inspections/${id}`);
}

export async function generateExplanation(
  inspectionId: string,
  detectionId: string,
  method: string,
): Promise<Explanation> {
  return apiFetch<Explanation>(
    `/api/v1/component1/inspections/${inspectionId}/explanations`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detection_id: detectionId, method }),
    },
  );
}

export async function getExplanationsForDetection(
  inspectionId: string,
  detectionId: string,
): Promise<Explanation[]> {
  return apiFetch<Explanation[]>(
    `/api/v1/component1/inspections/${inspectionId}/detections/${detectionId}/explanations`,
  );
}

export async function listInspections(params: {
  skip?: number;
  limit?: number;
  class_name?: string;
  from_date?: string;
  to_date?: string;
}): Promise<InspectionListResponse> {
  const qs = new URLSearchParams();
  if (params.skip != null) qs.set('skip', String(params.skip));
  if (params.limit != null) qs.set('limit', String(params.limit));
  if (params.class_name) qs.set('class_name', params.class_name);
  if (params.from_date) qs.set('from_date', params.from_date);
  if (params.to_date) qs.set('to_date', params.to_date);
  return apiFetch<InspectionListResponse>(`/api/v1/component1/inspections?${qs.toString()}`);
}
