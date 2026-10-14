import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import XaiComparison from '../components/XaiComparison';
import { decodeHeatmap, defectCrop, heatmapPixels } from '../utils/heatmap';
import type { Detection, Explanation } from '../types/component1';

vi.mock('../utils/heatmap', async importOriginal => ({ ...await importOriginal<typeof import('../utils/heatmap')>(), heatmapImage: () => 'data:image/png;base64,raw-fixture' }));
const detection: Detection = { id:'d1', detection_order:0, class_id:6, class_name:'stain', confidence:.8, pre_nms_index:42, bbox:{x_min:100,y_min:50,x_max:300,y_max:150} };
const explanation: Explanation = { id:'e1', detection_id:'d1', xai_method:'grad-cam', status:'completed', outcome:'success', overlay_url:'/composite.png', raw_heatmap_url:null, model_version:'test', target_layer:null, settings:null, error_message:null, inference_time_ms:null, generated_at:null, created_at:'2026-10-16' };
function npy(values: number[], width: number, height: number, bigEndian = false) {
  const header = new TextEncoder().encode(`{'descr': '${bigEndian ? '>' : '<'}f4', 'fortran_order': False, 'shape': (${height}, ${width}), }\n`);
  const buffer = new ArrayBuffer(10 + header.length + values.length * 4);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x93,78,85,77,80,89,1,0]);
  const view = new DataView(buffer); view.setUint16(8,header.length,true); bytes.set(header,10);
  values.forEach((v,i) => view.setFloat32(10+header.length+i*4,v,!bigEndian));
  return buffer;
}
afterEach(() => vi.unstubAllGlobals());

describe('Genuine heatmap decoding and display mapping', () => {
  it.each([false,true])('preserves non-square coordinates and raw values (big endian %s)', big => {
    const map = decodeHeatmap(npy([0,2,4,6,8,10],3,2,big),3,2);
    expect([map.width,map.height]).toEqual([3,2]);
    expect([...map.values]).toEqual([0,2,4,6,8,10]);
    expect([map.min,map.max]).toEqual([0,10]);
    const pixels=heatmapPixels(map);
    expect([...pixels.slice(0,4)]).toEqual([0,0,128,255]);
    expect([...pixels.slice(-4)]).toEqual([128,0,0,255]);
    expect([...map.values]).toEqual([0,2,4,6,8,10]);
  });
  it('rejects mismatched coordinates, invalid arrays and degenerate data', () => {
    expect(() => decodeHeatmap(npy([0,1],2,1),1,2)).toThrow(/dimensions/);
    expect(() => decodeHeatmap(npy([0,NaN],2,1),2,1)).toThrow(/non-finite/);
    expect(() => decodeHeatmap(npy([1,1],2,1),2,1)).toThrow(/range/);
    expect(() => decodeHeatmap(new ArrayBuffer(4),2,1)).toThrow(/Invalid/);
  });
  it('clamps crop context at image edges without changing the actual box', () => {
    const box={x_min:0,y_min:0,x_max:20,y_max:10};
    expect(defectCrop(box,100,50)).toEqual({x_min:0,y_min:0,x_max:23,y_max:11.5});
    expect(box).toEqual({x_min:0,y_min:0,x_max:20,y_max:10});
  });
});

describe('XAI comparison modes', () => {
  it('draws the same actual box over both images and crops using shared coordinates', async () => {
    const user=userEvent.setup();
    const {container}=render(<XaiComparison explanation={explanation} imageUrl="/original.png" imageWidth={1000} imageHeight={500} detection={detection} />);
    const boxes=screen.getAllByRole('img',{name:'Selected stain bounding box'});
    expect(boxes).toHaveLength(2);
    boxes.forEach(box=>expect(box).toHaveStyle({left:'10%',top:'10%',width:'20%',height:'20%'}));
    expect(screen.getByAltText('XAI heatmap overlay')).toHaveStyle({opacity:'1'});
    expect(screen.getByRole('button',{name:'Heatmap Only'})).toBeDisabled();
    await user.click(screen.getByRole('button',{name:'Defect Zoom'}));
    container.querySelectorAll('.xai-frame').forEach(frame=>expect(frame).toHaveAttribute('data-crop','70,35,330,165'));
    expect(screen.getByAltText('XAI heatmap overlay')).toHaveAttribute('src','/composite.png');
    boxes.forEach(box=>expect(box).toHaveStyle({left:'10%',top:'10%',width:'20%',height:'20%'}));
    await user.click(screen.getByRole('button',{name:'Original'}));
    expect(screen.queryByAltText('XAI heatmap overlay')).not.toBeInTheDocument();
    expect(screen.getByLabelText(/comparison opacity/i)).toBeDisabled();
    await user.click(screen.getByRole('button',{name:'Overlay'}));
    fireEvent.change(screen.getByLabelText(/comparison opacity/i),{target:{value:'0.4'}});
    expect(screen.getByAltText('XAI heatmap overlay')).toHaveStyle({opacity:'0.4'});
    expect(screen.getByLabelText('Heatmap color legend')).toHaveTextContent('Low / blue (0)');
  });
  it('loads only the genuine raw URL on demand and reuses it between modes', async () => {
    const fetcher=vi.fn().mockResolvedValue({ok:true,arrayBuffer:async()=>npy([0,1],2,1)});
    vi.stubGlobal('fetch',fetcher);
    const user=userEvent.setup();
    render(<XaiComparison explanation={{...explanation,raw_heatmap_url:'/raw.npy'}} imageUrl="/original.png" imageWidth={2} imageHeight={1} />);
    expect(fetcher).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button',{name:'Heatmap Only'}));
    expect(await screen.findByAltText('Raw heatmap visualization')).toHaveAttribute('src','data:image/png;base64,raw-fixture');
    expect(screen.queryByAltText('XAI heatmap overlay')).not.toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledWith('/raw.npy',expect.objectContaining({signal:expect.any(AbortSignal)}));
    await user.click(screen.getByRole('button',{name:'Overlay'}));
    await user.click(screen.getByRole('button',{name:'Heatmap Only'}));
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('fails clearly when raw data dimensions mismatch, without showing a composite as raw', async () => {
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,arrayBuffer:async()=>npy([0,1],2,1)}));
    const user=userEvent.setup();
    render(<XaiComparison explanation={{...explanation,raw_heatmap_url:'/raw.npy'}} imageUrl="/original.png" imageWidth={20} imageHeight={10} />);
    await user.click(screen.getByRole('button',{name:'Heatmap Only'}));
    await waitFor(()=>expect(screen.getByRole('alert')).toHaveTextContent('dimensions'));
    expect(screen.queryByAltText('Raw heatmap visualization')).not.toBeInTheDocument();
    expect(screen.queryByAltText('XAI heatmap overlay')).not.toBeInTheDocument();
  });
});


it('does not label the original as an overlay when only a raw artifact exists', () => {
  render(<XaiComparison explanation={{...explanation,overlay_url:null,raw_heatmap_url:'/raw.npy'}} imageUrl="/original.png" imageWidth={1000} imageHeight={500} detection={detection} />);
  expect(screen.getByRole('button',{name:'Original'})).toHaveAttribute('aria-pressed','true');
  expect(screen.getByRole('button',{name:'Overlay'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'Defect Zoom'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'Heatmap Only'})).toBeEnabled();
});
