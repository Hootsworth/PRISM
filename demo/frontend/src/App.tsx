import React, { useState, useEffect, useRef } from 'react';
import {
  Zap,
  Download,
  Upload,
  RefreshCw,
  Image as ImageIcon
} from 'lucide-react';

interface Sample {
  id: string;
  instance_id: string;
  pair_index: string;
  camera: string;
  iso: number;
  shutter: string;
  cfa: string;
  height: number;
  width: number;
  raw_noisy_path: string;
  raw_metadata_path: string;
  gt_srgb_path: string | null;
  noisy_srgb_path: string | null;
  split: string;
}

interface ProcessResult {
  process_id: string;
  pipeline: string;
  elapsed_seconds: number;
  stage_timings: Record<string, number>;
  stats: Record<string, any>;
  metrics: { psnr_db: number; ssim: number; mse: number } | null;
  preview_image: string;
  baseline_image: string;
  unprocessed_image: string;
  heatmap_image: string | null;
  overlay_image: string | null;
  download_url: string;
  device: string;
  checkpoint: string | null;
}

export default function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null);
  const [deviceFilter, setDeviceFilter] = useState<'ALL' | 'IP' | 'GP'>('ALL');

  // Upload
  const [uploadedFile, setUploadedFile] = useState<any | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [inputTab, setInputTab] = useState<'samples' | 'upload'>('samples');

  // Processing state
  const [processing, setProcessing] = useState<boolean>(false);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Comparison & View state
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [viewMode, setViewMode] = useState<'slider' | 'side-by-side' | 'three-way' | 'enhanced'>('slider');
  const [comparisonBase, setComparisonBase] = useState<'unprocessed' | 'baseline'>('unprocessed');
  const [showAiMap, setShowAiMap] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef<boolean>(false);

  useEffect(() => {
    fetch('/api/samples')
      .then(res => res.json())
      .then(data => {
        if (data.samples && data.samples.length > 0) {
          setSamples(data.samples);
          setSelectedSample(data.samples[0]);
        }
      })
      .catch(err => setError('Could not load sample photos: ' + err.message));
  }, []);

  const handleMouseDown = () => { isDragging.current = true; };
  const handleMouseUp = () => { isDragging.current = false; };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  };
  const handleTouchMove = (e: React.TouchEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.touches[0].clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  };

  const handleProcess = async () => {
    setProcessing(true);
    setError(null);
    try {
      const payload = {
        source_type: inputTab === 'samples' ? 'sample' : 'upload',
        source_id: inputTab === 'samples' ? selectedSample?.id : uploadedFile?.file_id,
        pipeline: 'B2',
        budget: 0.25,
        tone_map: 'reinhard',
        max_preview_dim: 2048,
        sharpness: 0.85,
      };

      const res = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Processing failed');
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('raw_file', files[0]);
    if (files.length > 1) {
      formData.append('meta_file', files[1]);
    }

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setUploadedFile(data);
        setInputTab('upload');
      }
    } catch (err: any) {
      setError('Upload failed: ' + err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#fafafa', color: '#18181b' }}>
      {/* Header */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '16px 32px',
        background: '#ffffff',
        borderBottom: '1px solid #e4e4e7',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '10px',
            background: '#18181b',
            color: '#ffffff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 700,
            fontSize: '15px'
          }}>
            P
          </div>
          <div>
            <h1 style={{ fontSize: '16px', fontWeight: 700, letterSpacing: '-0.02em', margin: 0, lineHeight: 1.2 }}>PRISM</h1>
            <p style={{ fontSize: '12px', color: '#71717a', margin: 0 }}>Smart Camera Image Processing</p>
          </div>
        </div>

        {/* Action / Download */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {result && (
            <a
              href={result.download_url}
              download
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 16px',
                borderRadius: '10px',
                background: '#18181b',
                color: '#ffffff',
                fontSize: '13px',
                fontWeight: 600,
                textDecoration: 'none',
                boxShadow: '0 2px 4px rgba(0,0,0,0.06)'
              }}>
              <Download size={15} /> Save Photo
            </a>
          )}
        </div>
      </header>

      {/* Main Content */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Sidebar Controls */}
        <aside style={{
          width: '380px',
          background: '#ffffff',
          borderRight: '1px solid #e4e4e7',
          padding: '28px',
          display: 'flex',
          flexDirection: 'column',
          gap: '24px',
          overflowY: 'auto'
        }}>
          {/* Section 1: Choose Photo */}
          <div>
            <div style={{ fontSize: '11px', fontWeight: 700, color: '#a1a1aa', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '10px' }}>
              1. Select Photo
            </div>
            
            {/* Tabs */}
            <div style={{ display: 'flex', gap: '6px', padding: '4px', background: '#f4f4f5', borderRadius: '12px', marginBottom: '12px' }}>
              <button
                onClick={() => setInputTab('samples')}
                style={{
                  flex: 1,
                  padding: '8px',
                  borderRadius: '8px',
                  border: 'none',
                  background: inputTab === 'samples' ? '#ffffff' : 'transparent',
                  color: inputTab === 'samples' ? '#18181b' : '#71717a',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: 'pointer',
                  boxShadow: inputTab === 'samples' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
                }}>
                Sample Photos
              </button>
              <button
                onClick={() => setInputTab('upload')}
                style={{
                  flex: 1,
                  padding: '8px',
                  borderRadius: '8px',
                  border: 'none',
                  background: inputTab === 'upload' ? '#ffffff' : 'transparent',
                  color: inputTab === 'upload' ? '#18181b' : '#71717a',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: 'pointer',
                  boxShadow: inputTab === 'upload' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
                }}>
                Upload RAW
              </button>
            </div>

            {inputTab === 'samples' ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {/* Camera Filter Pills */}
                <div style={{ display: 'flex', gap: '6px' }}>
                  <button
                    onClick={() => {
                      setDeviceFilter('ALL');
                    }}
                    style={{
                      flex: 1,
                      padding: '6px 8px',
                      borderRadius: '8px',
                      border: '1px solid ' + (deviceFilter === 'ALL' ? '#18181b' : '#e4e4e7'),
                      background: deviceFilter === 'ALL' ? '#18181b' : '#ffffff',
                      color: deviceFilter === 'ALL' ? '#ffffff' : '#52525b',
                      fontSize: '11px',
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}>
                    All ({samples.length})
                  </button>
                  <button
                    onClick={() => {
                      setDeviceFilter('IP');
                      const firstIp = samples.find(s => s.camera === 'IP');
                      if (firstIp) setSelectedSample(firstIp);
                    }}
                    style={{
                      flex: 1.2,
                      padding: '6px 8px',
                      borderRadius: '8px',
                      border: '1px solid ' + (deviceFilter === 'IP' ? '#18181b' : '#e4e4e7'),
                      background: deviceFilter === 'IP' ? '#18181b' : '#ffffff',
                      color: deviceFilter === 'IP' ? '#ffffff' : '#52525b',
                      fontSize: '11px',
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}>
                    Apple iPhone ({samples.filter(s => s.camera === 'IP').length})
                  </button>
                  <button
                    onClick={() => {
                      setDeviceFilter('GP');
                      const firstGp = samples.find(s => s.camera === 'GP');
                      if (firstGp) setSelectedSample(firstGp);
                    }}
                    style={{
                      flex: 1.2,
                      padding: '6px 8px',
                      borderRadius: '8px',
                      border: '1px solid ' + (deviceFilter === 'GP' ? '#18181b' : '#e4e4e7'),
                      background: deviceFilter === 'GP' ? '#18181b' : '#ffffff',
                      color: deviceFilter === 'GP' ? '#ffffff' : '#52525b',
                      fontSize: '11px',
                      fontWeight: 600,
                      cursor: 'pointer'
                    }}>
                    Google Pixel ({samples.filter(s => s.camera === 'GP').length})
                  </button>
                </div>

                <select
                  value={selectedSample?.id || ''}
                  onChange={e => {
                    const found = samples.find(s => s.id === e.target.value);
                    if (found) setSelectedSample(found);
                  }}
                  style={{
                    width: '100%',
                    padding: '10px 14px',
                    borderRadius: '12px',
                    border: '1px solid #e4e4e7',
                    background: '#ffffff',
                    fontSize: '13px',
                    color: '#18181b',
                    outline: 'none',
                    fontWeight: 500,
                    cursor: 'pointer'
                  }}>
                  {samples
                    .filter(s => deviceFilter === 'ALL' || s.camera === deviceFilter)
                    .map((s, idx) => (
                      <option key={s.id} value={s.id}>
                        {s.camera === 'IP' ? 'Apple iPhone' : 'Google Pixel'} • ISO {s.iso} • Sample #{idx + 1}
                      </option>
                    ))}
                </select>

                {selectedSample && (
                  <div style={{ padding: '12px 14px', borderRadius: '12px', background: '#f4f4f5', fontSize: '12px', color: '#52525b', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Device: <b>{selectedSample.camera === 'IP' ? 'Apple iPhone' : (selectedSample.camera === 'GP' ? 'Google Pixel' : selectedSample.camera)}</b></span>
                    <span>ISO: <b>{selectedSample.iso}</b></span>
                    <span>Format: <b>{selectedSample.cfa}</b></span>
                  </div>
                )}
              </div>
            ) : (
              <div>
                <label style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  padding: '24px',
                  borderRadius: '14px',
                  border: '1px dashed #d4d4d8',
                  background: '#fafafa',
                  cursor: 'pointer',
                  textAlign: 'center'
                }}>
                  <Upload size={20} color="#71717a" style={{ marginBottom: '8px' }} />
                  <span style={{ fontSize: '13px', fontWeight: 600 }}>Choose RAW file</span>
                  <span style={{ fontSize: '11px', color: '#a1a1aa', marginTop: '2px' }}>.MAT files supported</span>
                  <input type="file" multiple accept=".mat,.MAT" onChange={handleFileUpload} style={{ display: 'none' }} />
                </label>
                {uploading && <div style={{ fontSize: '12px', color: '#71717a', marginTop: '8px', textAlign: 'center' }}>Reading photo...</div>}
                {uploadedFile && (
                  <div style={{ marginTop: '10px', padding: '12px', borderRadius: '10px', background: '#f4f4f5', fontSize: '12px' }}>
                    <b>{uploadedFile.filename}</b> ({uploadedFile.width}×{uploadedFile.height})
                  </div>
                )}
              </div>
            )}
          </div>

          {/* PRISM Hybrid Engine Info Card */}
          <div style={{
            padding: '16px 18px',
            borderRadius: '14px',
            background: '#f4f4f5',
            border: '1px solid #e4e4e7'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#22c55e' }} />
              <span style={{ fontSize: '12px', fontWeight: 700, color: '#18181b' }}>PRISM Smart Hybrid Engine</span>
            </div>
            <p style={{ fontSize: '12px', color: '#52525b', margin: 0, lineHeight: 1.5 }}>
              Combines ultra-fast camera physics (Standard) with targeted deep learning (AI) at the optimal budget for crisp details and maximum battery efficiency.
            </p>
          </div>

          {/* Edge Sharpness & Acutance Status (Hardcoded 85% Auto-Optimized) */}
          <div style={{
            padding: '16px 18px',
            borderRadius: '14px',
            background: '#ffffff',
            border: '1px solid #e4e4e7',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div>
              <div style={{ fontSize: '11px', fontWeight: 700, color: '#18181b', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Edge Acutance & Sharpness
              </div>
              <div style={{ fontSize: '12px', color: '#71717a', marginTop: '2px' }}>
                Auto-optimized high micro-contrast
              </div>
            </div>
            <div style={{
              background: '#eff6ff',
              color: '#2563eb',
              fontWeight: 700,
              fontSize: '12px',
              padding: '5px 11px',
              borderRadius: '8px',
              border: '1px solid #dbeafe'
            }}>
              85% Sharp
            </div>
          </div>

          {/* Process Button */}
          <div style={{ marginTop: 'auto' }}>
            {error && (
              <div style={{
                padding: '10px 14px',
                borderRadius: '10px',
                background: '#fef2f2',
                border: '1px solid #fecaca',
                color: '#b91c1c',
                fontSize: '12px',
                marginBottom: '12px'
              }}>
                {error}
              </div>
            )}
            <button
              onClick={handleProcess}
              disabled={processing}
              style={{
                width: '100%',
                padding: '14px',
                borderRadius: '12px',
                border: 'none',
                background: processing ? '#a1a1aa' : '#18181b',
                color: '#ffffff',
                fontSize: '14px',
                fontWeight: 600,
                cursor: processing ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                transition: 'all 0.15s ease'
              }}>
              {processing ? (
                <>
                  <RefreshCw size={16} className="animate-spin" />
                  <span>Processing photo...</span>
                </>
              ) : (
                <>
                  <Zap size={16} />
                  <span>Process Photo</span>
                </>
              )}
            </button>
          </div>
        </aside>

        {/* Center Viewer Area */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#f4f4f5', overflow: 'hidden' }}>
          {/* Viewer Toolbar */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '12px 28px',
            background: '#ffffff',
            borderBottom: '1px solid #e4e4e7',
            flexWrap: 'wrap',
            gap: '12px'
          }}>
            {/* View Mode Switcher */}
            <div style={{ display: 'flex', gap: '6px', background: '#f4f4f5', padding: '3px', borderRadius: '10px' }}>
              <button
                onClick={() => setViewMode('slider')}
                style={{
                  padding: '6px 12px',
                  borderRadius: '7px',
                  border: 'none',
                  background: viewMode === 'slider' ? '#ffffff' : 'transparent',
                  color: viewMode === 'slider' ? '#18181b' : '#71717a',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: 'pointer',
                  boxShadow: viewMode === 'slider' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
                }}>
                Split Slider
              </button>
              <button
                onClick={() => setViewMode('side-by-side')}
                style={{
                  padding: '6px 12px',
                  borderRadius: '7px',
                  border: 'none',
                  background: viewMode === 'side-by-side' ? '#ffffff' : 'transparent',
                  color: viewMode === 'side-by-side' ? '#18181b' : '#71717a',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: 'pointer',
                  boxShadow: viewMode === 'side-by-side' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
                }}>
                Side-by-Side
              </button>
              <button
                onClick={() => setViewMode('three-way')}
                style={{
                  padding: '6px 12px',
                  borderRadius: '7px',
                  border: 'none',
                  background: viewMode === 'three-way' ? '#ffffff' : 'transparent',
                  color: viewMode === 'three-way' ? '#18181b' : '#71717a',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: 'pointer',
                  boxShadow: viewMode === 'three-way' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
                }}>
                3-Way Comparison
              </button>
              <button
                onClick={() => setViewMode('enhanced')}
                style={{
                  padding: '6px 12px',
                  borderRadius: '7px',
                  border: 'none',
                  background: viewMode === 'enhanced' ? '#ffffff' : 'transparent',
                  color: viewMode === 'enhanced' ? '#18181b' : '#71717a',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: 'pointer',
                  boxShadow: viewMode === 'enhanced' ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
                }}>
                PRISM Output Only
              </button>
            </div>

            {/* Reference Image Comparison Controls */}
            {result && viewMode !== 'enhanced' && viewMode !== 'three-way' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#71717a', fontWeight: 600 }}>Compare PRISM against:</span>
                <div style={{ display: 'flex', gap: '4px', background: '#f4f4f5', padding: '3px', borderRadius: '8px' }}>
                  <button
                    onClick={() => setComparisonBase('unprocessed')}
                    style={{
                      padding: '5px 12px',
                      borderRadius: '6px',
                      border: 'none',
                      background: comparisonBase === 'unprocessed' ? '#18181b' : 'transparent',
                      color: comparisonBase === 'unprocessed' ? '#ffffff' : '#71717a',
                      fontWeight: 600,
                      fontSize: '11px',
                      cursor: 'pointer',
                      boxShadow: comparisonBase === 'unprocessed' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none'
                    }}>
                    Non-Processed (Raw Noisy)
                  </button>
                  <button
                    onClick={() => setComparisonBase('baseline')}
                    style={{
                      padding: '5px 12px',
                      borderRadius: '6px',
                      border: 'none',
                      background: comparisonBase === 'baseline' ? '#18181b' : 'transparent',
                      color: comparisonBase === 'baseline' ? '#ffffff' : '#71717a',
                      fontWeight: 600,
                      fontSize: '11px',
                      cursor: 'pointer',
                      boxShadow: comparisonBase === 'baseline' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none'
                    }}>
                    Standard Camera (B0)
                  </button>
                </div>
              </div>
            )}

            {/* AI Focus Map Toggle */}
            {result?.heatmap_image && (
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#18181b', fontWeight: 500, cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showAiMap}
                  onChange={e => setShowAiMap(e.target.checked)}
                  style={{ accentColor: '#18181b', cursor: 'pointer' }}
                />
                <span>Show AI Focus Heatmap</span>
              </label>
            )}
          </div>

          {/* Interactive Image Display */}
          <div
            ref={containerRef}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onTouchMove={handleTouchMove}
            style={{
              flex: 1,
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '24px',
              overflow: 'hidden'
            }}>
            {result ? (
              viewMode === 'three-way' ? (
                /* 3-Way Comprehensive Comparison */
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', width: '100%', height: '100%', maxHeight: '74vh', alignItems: 'center' }}>
                  {/* Panel 1: Non-processed RAW */}
                  <div style={{ position: 'relative', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', borderRadius: '14px', overflow: 'hidden' }}>
                    <img
                      src={result.unprocessed_image || result.baseline_image}
                      alt="Non-Processed RAW"
                      style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                    />
                    <div style={{ position: 'absolute', top: 12, left: 12, background: 'rgba(239, 68, 68, 0.9)', backdropFilter: 'blur(8px)', color: '#fff', padding: '4px 10px', borderRadius: '8px', fontSize: '11px', fontWeight: 700 }}>
                      1. Non-Processed (Raw Noisy Sensor)
                    </div>
                  </div>

                  {/* Panel 2: Standard Camera Baseline (B0) */}
                  <div style={{ position: 'relative', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', borderRadius: '14px', overflow: 'hidden' }}>
                    <img
                      src={result.baseline_image}
                      alt="Standard Camera B0"
                      style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                    />
                    <div style={{ position: 'absolute', top: 12, left: 12, background: 'rgba(24, 24, 27, 0.85)', backdropFilter: 'blur(8px)', color: '#fff', padding: '4px 10px', borderRadius: '8px', fontSize: '11px', fontWeight: 600 }}>
                      2. Standard Camera (B0 Baseline)
                    </div>
                  </div>

                  {/* Panel 3: PRISM Smart Camera (B2) */}
                  <div style={{ position: 'relative', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', borderRadius: '14px', overflow: 'hidden' }}>
                    <img
                      src={result.preview_image}
                      alt="PRISM Smart Camera"
                      style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                    />
                    <div style={{ position: 'absolute', top: 12, right: 12, background: '#2563eb', color: '#fff', padding: '4px 10px', borderRadius: '8px', fontSize: '11px', fontWeight: 700 }}>
                      3. PRISM Smart Camera (85% Sharp)
                    </div>
                    {showAiMap && result.heatmap_image && (
                      <img src={result.heatmap_image} alt="AI Map" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain', opacity: 0.65, pointerEvents: 'none' }} />
                    )}
                  </div>
                </div>
              ) : viewMode === 'side-by-side' ? (
                /* Side-by-Side Dual View */
                <div style={{ display: 'flex', gap: '20px', width: '100%', height: '100%', maxHeight: '74vh', alignItems: 'center', justifyContent: 'center' }}>
                  {/* Left: Reference (Non-processed or B0) */}
                  <div style={{ flex: 1, position: 'relative', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', borderRadius: '14px', overflow: 'hidden' }}>
                    <img
                      src={comparisonBase === 'unprocessed' ? (result.unprocessed_image || result.baseline_image) : result.baseline_image}
                      alt="Reference"
                      style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                    />
                    <div style={{
                      position: 'absolute',
                      top: 12,
                      left: 12,
                      background: comparisonBase === 'unprocessed' ? 'rgba(239, 68, 68, 0.9)' : 'rgba(24,24,27,0.85)',
                      backdropFilter: 'blur(8px)',
                      color: '#fff',
                      padding: '4px 10px',
                      borderRadius: '8px',
                      fontSize: '11px',
                      fontWeight: 600
                    }}>
                      {comparisonBase === 'unprocessed' ? 'Non-Processed (Raw Noisy Capture)' : 'Standard Camera (Baseline B0)'}
                    </div>
                  </div>
                  {/* Right: PRISM Result */}
                  <div style={{ flex: 1, position: 'relative', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', borderRadius: '14px', overflow: 'hidden' }}>
                    <img src={result.preview_image} alt="Enhanced" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                    <div style={{ position: 'absolute', top: 12, right: 12, background: '#2563eb', color: '#fff', padding: '4px 10px', borderRadius: '8px', fontSize: '11px', fontWeight: 700 }}>
                      PRISM Smart Camera (85% Sharp)
                    </div>
                    {showAiMap && result.heatmap_image && (
                      <img src={result.heatmap_image} alt="AI Map" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain', opacity: 0.65, pointerEvents: 'none' }} />
                    )}
                  </div>
                </div>
              ) : viewMode === 'enhanced' ? (
                /* AI Output Only */
                <div style={{ position: 'relative', maxWidth: '92%', maxHeight: '74vh', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', borderRadius: '14px', overflow: 'hidden' }}>
                  <img src={result.preview_image} alt="Enhanced" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                  <div style={{ position: 'absolute', top: 12, right: 12, background: '#2563eb', color: '#fff', padding: '4px 10px', borderRadius: '8px', fontSize: '11px', fontWeight: 700 }}>
                    PRISM Smart Camera (85% Sharp)
                  </div>
                  {showAiMap && result.heatmap_image && (
                    <img src={result.heatmap_image} alt="AI Map" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain', opacity: 0.65, pointerEvents: 'none' }} />
                  )}
                </div>
              ) : (
                /* Split Comparison Slider */
                <div className="slider-stage" style={{ maxWidth: '92%', maxHeight: '72vh', width: '100%', position: 'relative' }}>
                  {/* Left: Selected Reference Image */}
                  <img
                    src={comparisonBase === 'unprocessed' ? (result.unprocessed_image || result.baseline_image) : result.baseline_image}
                    alt="Reference"
                    className="slider-img-base"
                  />
                  <div style={{
                    position: 'absolute',
                    top: 16,
                    left: 16,
                    background: comparisonBase === 'unprocessed' ? 'rgba(239, 68, 68, 0.9)' : 'rgba(24, 24, 27, 0.85)',
                    backdropFilter: 'blur(8px)',
                    color: '#ffffff',
                    padding: '4px 10px',
                    borderRadius: '8px',
                    fontSize: '11px',
                    fontWeight: 600,
                    pointerEvents: 'none'
                  }}>
                    {comparisonBase === 'unprocessed' ? 'Non-Processed (Raw Noisy Sensor)' : 'Standard Camera (Baseline B0)'}
                  </div>

                  {/* Right: PRISM Smart Camera Enhanced Photo */}
                  <div
                    className="slider-img-overlay"
                    style={{ clipPath: `polygon(${sliderPos}% 0, 100% 0, 100% 100%, ${sliderPos}% 100%)` }}>
                    <img src={result.preview_image} alt="Enhanced" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                    <div style={{
                      position: 'absolute',
                      top: 16,
                      right: 16,
                      background: '#2563eb',
                      color: '#ffffff',
                      padding: '4px 10px',
                      borderRadius: '8px',
                      fontSize: '11px',
                      fontWeight: 700,
                      pointerEvents: 'none'
                    }}>
                      PRISM Smart Camera (85% Sharp)
                    </div>

                    {/* AI Heatmap layer */}
                    {showAiMap && result.heatmap_image && (
                      <img
                        src={result.heatmap_image}
                        alt="AI Map"
                        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain', opacity: 0.65, pointerEvents: 'none' }}
                      />
                    )}
                  </div>

                  {/* Divider Handle */}
                  <div className="slider-divider" style={{ left: `${sliderPos}%` }}>
                    <div
                      className="slider-handle"
                      onMouseDown={handleMouseDown}
                      onTouchStart={handleMouseDown}>
                      ↔
                    </div>
                  </div>
                </div>
              )
            ) : (
              /* Empty state */
              <div style={{ textAlign: 'center', maxWidth: '360px' }}>
                <div style={{
                  width: '56px',
                  height: '56px',
                  borderRadius: '16px',
                  background: '#ffffff',
                  border: '1px solid #e4e4e7',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 16px',
                  boxShadow: '0 2px 6px rgba(0,0,0,0.04)'
                }}>
                  <ImageIcon size={24} color="#a1a1aa" />
                </div>
                <h3 style={{ fontSize: '15px', fontWeight: 600, margin: '0 0 6px 0', color: '#18181b' }}>Photo Ready</h3>
                <p style={{ fontSize: '13px', color: '#71717a', margin: 0, lineHeight: 1.5 }}>
                  Choose a photo on the left, select a processing mode, and click Process Photo.
                </p>
              </div>
            )}
          </div>

          {/* Simple Bottom Metrics Card */}
          {result && (
            <footer style={{
              background: '#ffffff',
              borderTop: '1px solid #e4e4e7',
              padding: '16px 32px',
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '16px'
            }}>
              {/* Metric 1 */}
              <div style={{ padding: '12px 16px', background: '#fcfcfd', borderRadius: '12px', border: '1px solid #e4e4e7' }}>
                <div style={{ fontSize: '11px', color: '#71717a', fontWeight: 600 }}>Processing Time</div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#18181b', marginTop: '2px' }}>
                  {result.elapsed_seconds < 1 ? `${(result.elapsed_seconds * 1000).toFixed(0)} ms` : `${result.elapsed_seconds.toFixed(2)} s`}
                </div>
              </div>

              {/* Metric 2 */}
              <div style={{ padding: '12px 16px', background: '#fcfcfd', borderRadius: '12px', border: '1px solid #e4e4e7' }}>
                <div style={{ fontSize: '11px', color: '#71717a', fontWeight: 600 }}>PSNR vs. Ground Truth</div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#18181b', marginTop: '2px' }}>
                  {result.metrics ? `${result.metrics.psnr_db.toFixed(1)} dB` : 'N/A'}
                </div>
              </div>

              {/* Metric 3 */}
              <div style={{ padding: '12px 16px', background: '#fcfcfd', borderRadius: '12px', border: '1px solid #e4e4e7' }}>
                <div style={{ fontSize: '11px', color: '#71717a', fontWeight: 600 }}>AI Applied</div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#18181b', marginTop: '2px' }}>
                  {result.pipeline === 'B2' ? `${result.stats.refinement_percentage ?? 0}% of photo` : (result.pipeline === 'B1' ? '100% (Full AI)' : '0% (Standard)')}
                </div>
              </div>

              {/* Metric 4 */}
              <div style={{ padding: '12px 16px', background: '#fcfcfd', borderRadius: '12px', border: '1px solid #e4e4e7' }}>
                <div style={{ fontSize: '11px', color: '#71717a', fontWeight: 600 }}>Tiles Left on B0</div>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#18181b', marginTop: '2px' }}>
                  {result.pipeline === 'B2'
                    ? `${Math.max(0, 100 - (result.stats.refinement_percentage || 0)).toFixed(0)}%`
                    : (result.pipeline === 'B0' ? '100%' : '0%')}
                </div>
              </div>
            </footer>
          )}
        </main>
      </div>
    </div>
  );
}
