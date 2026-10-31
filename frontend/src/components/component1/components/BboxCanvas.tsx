import { useEffect, useRef, useState } from 'react';
import { artifactUrl } from '../../../shared/api/client';
import type { Detection } from '../types/component1';

const CLASS_COLORS: Record<string, string> = {
  baekra: '#e6194b',
  'color issues': '#3cb44b',
  contamination: '#ffe119',
  cut: '#4363d8',
  'gray stitch': '#f58231',
  selvet: '#911eb4',
  stain: '#42d4f4',
};

interface Props {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  detections: Detection[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

export default function BboxCanvas({
  imageUrl,
  imageWidth,
  imageHeight,
  detections,
  selectedId,
  onSelect,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [displaySize, setDisplaySize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect;
      const scale = width / imageWidth;
      setDisplaySize({ w: width, h: imageHeight * scale });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [imageWidth, imageHeight]);

  const scaleX = displaySize.w / imageWidth;
  const scaleY = displaySize.h / imageHeight;

  return (
    <div
      ref={containerRef}
      style={{ position: 'relative', width: '100%', overflow: 'hidden' }}
    >
      <img
        src={artifactUrl(imageUrl)}
        alt="Inspected fabric"
        style={{ width: '100%', display: 'block' }}
        draggable={false}
      />
      {displaySize.w > 0 &&
        detections.map((d) => {
          const isSelected = d.id === selectedId;
          const color = CLASS_COLORS[d.class_name] ?? '#ffffff';
          return (
            <button
              type="button"
              aria-label={`Select ${d.class_name}, ${(d.confidence * 100).toFixed(1)}% confidence`}
              aria-pressed={isSelected}
              key={d.id}
              onClick={() => onSelect(isSelected ? null : d.id)}
              title={`${d.class_name} ${(d.confidence * 100).toFixed(1)}%`}
              style={{
                position: 'absolute',
                padding: 0, minHeight: 0, borderRadius: 0,
                left: d.bbox.x_min * scaleX,
                top: d.bbox.y_min * scaleY,
                width: (d.bbox.x_max - d.bbox.x_min) * scaleX,
                height: (d.bbox.y_max - d.bbox.y_min) * scaleY,
                border: `${isSelected ? 3 : 2}px solid ${color}`,
                backgroundColor: isSelected
                  ? `${color}22`
                  : 'transparent',
                cursor: 'pointer',
                boxSizing: 'border-box',
              }}
            >
              <span
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  fontSize: 11,
                  backgroundColor: '#081321',
                  color: '#fff',
                  padding: '0 4px',
                  whiteSpace: 'nowrap',
                  fontWeight: isSelected ? 700 : 400,
                }}
              >
                {d.class_name} {(d.confidence * 100).toFixed(1)}%
              </span>
            </button>
          );
        })}
    </div>
  );
}
