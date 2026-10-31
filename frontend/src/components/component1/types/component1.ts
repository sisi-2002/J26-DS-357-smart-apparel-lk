export interface Bbox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface Detection {
  id: string;
  detection_order: number;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: Bbox;
  pre_nms_index: number;
}

export interface InspectionSummary {
  total_defects: number;
  classes_found: string[];
}

export interface ModelIdentity {
  model_version: string;
  model_weights_hash: string | null;
  device: string | null;
  confidence_threshold: number;
  iou_threshold: number;
  imgsz: number;
  /**
   * Backend-resolved experiment identity from this inspection's persisted
   * checkpoint SHA256 and an authoritative frozen manifest. May also resolve
   * older records with a known full hash. Never infer identity when absent.
   */
  experiment_id?: string | null;
}

export interface Inspection {
  id: string;
  image_width: number;
  image_height: number;
  image_url: string;
  thumbnail_url: string | null;
  detected_at: string;
  inference_time_ms: number | null;
  summary: InspectionSummary;
  model: ModelIdentity;
  detections: Detection[];
  complexity_tier: string | null;
  complexity_score: number | null;
}

export interface InspectionListItem {
  id: string;
  detected_at: string;
  thumbnail_url: string | null;
  total_defects: number;
  defect_classes: string[];
  model_version: string;
}

export interface InspectionListResponse {
  total: number;
  skip: number;
  limit: number;
  items: InspectionListItem[];
}

export interface Explanation {
  id: string;
  detection_id: string;
  xai_method: string;
  status: string;
  outcome: string | null;
  overlay_url: string | null;
  raw_heatmap_url: string | null;
  model_version: string;
  target_layer: string | null;
  settings: Record<string, unknown> | null;
  error_message: string | null;
  inference_time_ms: number | null;
  generated_at: string | null;
  created_at: string;
}
