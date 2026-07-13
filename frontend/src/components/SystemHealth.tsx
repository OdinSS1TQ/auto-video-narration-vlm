import React, { useState, useEffect } from 'react';
import { ShieldCheck, AlertTriangle, XCircle, Server, Cpu, HardDrive, Sliders, RefreshCw, CheckCircle2 } from 'lucide-react';
import { api, HealthResponse, ConfigResponse } from '../api';

export default function SystemHealth() {
  const [activeTab, setActiveTab] = useState<'metrics' | 'config'>('metrics');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchHealth = async (showProgress = false) => {
    if (showProgress) setRefreshing(true);
    try {
      const data = await api.getHealth();
      setHealth(data);
      setError(null);
    } catch (err: any) {
      console.error('Error fetching health metrics:', err);
      setError(err.message || 'Failed to connect to the API server.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const fetchConfig = async () => {
    try {
      const data = await api.getConfig();
      setConfig(data);
    } catch (err: any) {
      console.error('Error fetching engine config:', err);
    }
  };

  // Poll health metrics every 5 seconds
  useEffect(() => {
    fetchHealth();
    const interval = setInterval(() => {
      fetchHealth();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Fetch configs when config tab becomes active
  useEffect(() => {
    if (activeTab === 'config') {
      fetchConfig();
    }
  }, [activeTab]);

  if (loading && !health) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#0d0d0f] text-[#e5e1e4]">
        <div className="flex flex-col items-center gap-3">
          <RefreshCw className="w-8 h-8 animate-spin text-[#adc6ff]" />
          <p className="text-sm font-mono">Loading system health telemetry...</p>
        </div>
      </div>
    );
  }

  // Determine status color/icon
  const isHealthy = health?.status === 'healthy';
  const isDegraded = health?.status === 'degraded';
  const statusLabel = isHealthy ? 'System Healthy' : isDegraded ? 'System Degraded' : 'System Unhealthy';
  const statusDesc = isHealthy
    ? 'All core engines are operational and processing jobs efficiently.'
    : isDegraded
      ? 'Some optional dependencies are missing (VRAM/GPU is unavailable). System falls back to API translation.'
      : 'Required core dependencies (FFmpeg) are missing. Dubbing operations are unavailable.';

  // Storage calculations
  const uploadsSize = health?.disk_usage?.uploads_mb || 0;
  const outputsSize = health?.disk_usage?.outputs_mb || 0;

  return (
    <div className="flex-1 flex flex-col min-h-screen bg-[#0d0d0f] text-[#e5e1e4] p-8 pb-16">

      {/* Page Title & Tab buttons */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-sans font-bold text-3xl tracking-tight text-[#e5e1e4]">System Status</h1>
            {refreshing && <RefreshCw className="w-4 h-4 animate-spin text-[#adc6ff]" />}
          </div>
          <p className="text-sm text-[#c2c6d6]">Real-time health metrics and engine configuration.</p>
        </div>

        {/* Tab Buttons */}
        <div className="flex bg-[#141416] p-1 rounded-md border border-[#1c1b1d]">
          <button
            onClick={() => setActiveTab('metrics')}
            className={`px-4 py-2 font-mono text-xs font-semibold rounded cursor-pointer transition ${activeTab === 'metrics'
                ? 'bg-[#3626ce] text-white shadow-sm'
                : 'text-[#c2c6d6] hover:text-[#e5e1e4]'
              }`}
          >
            Health Metrics
          </button>
          <button
            onClick={() => setActiveTab('config')}
            className={`px-4 py-2 font-mono text-xs font-semibold rounded cursor-pointer transition ${activeTab === 'config'
                ? 'bg-[#3626ce] text-white shadow-sm'
                : 'text-[#c2c6d6] hover:text-[#e5e1e4]'
              }`}
          >
            Configuration
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6 flex items-start gap-3">
          <XCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-semibold text-red-400">Connection Error</h4>
            <p className="text-xs text-[#c2c6d6] mt-1">{error}</p>
          </div>
        </div>
      )}

      {activeTab === 'metrics' && health ? (
        <div className="flex flex-col gap-6">
          {/* Header overall Card */}
          <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-6 flex items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${isHealthy
                  ? 'bg-emerald-500/15 border border-emerald-500/30 text-emerald-400'
                  : isDegraded
                    ? 'bg-amber-500/15 border border-amber-500/30 text-amber-400'
                    : 'bg-red-500/15 border border-red-500/30 text-red-400'
                }`}>
                {isHealthy ? (
                  <ShieldCheck className="w-6 h-6" />
                ) : isDegraded ? (
                  <AlertTriangle className="w-6 h-6" />
                ) : (
                  <XCircle className="w-6 h-6" />
                )}
              </div>
              <div>
                <h2 className="text-lg font-bold text-[#e5e1e4] mb-1">{statusLabel}</h2>
                <p className="text-xs text-[#c2c6d6]">{statusDesc}</p>
              </div>
            </div>
            <div className="text-right hidden sm:block font-mono">
              <span className="text-[10px] text-[#c2c6d6] uppercase tracking-wider block mb-1">VERSION</span>
              <span className="text-sm font-bold text-[#adc6ff]">{health.version}</span>
            </div>
          </div>

          {/* Bento Status Grid - 4 Columns */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">

            {/* Card A: FFmpeg */}
            <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-5 flex flex-col justify-between min-h-[140px]">
              <div className="flex items-center justify-between mb-4">
                <Server className="w-4 h-4 text-[#adc6ff]" />
                {health.dependencies.ffmpeg.available ? (
                  <span className="font-mono text-[9px] bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 text-emerald-400 rounded">
                    READY
                  </span>
                ) : (
                  <span className="font-mono text-[9px] bg-red-500/10 border border-red-500/20 px-2 py-0.5 text-red-400 rounded">
                    MISSING
                  </span>
                )}
              </div>
              <div>
                <span className="text-[10px] font-mono text-[#c2c6d6] block mb-1">Engine Module</span>
                <span className="text-base font-bold text-[#e5e1e4] block">FFmpeg</span>
                <span className="font-mono text-[10px] text-[#c2c6d6] opacity-80 truncate block">
                  {health.dependencies.ffmpeg.version ? `v${health.dependencies.ffmpeg.version.substring(0, 20)}` : 'Unavailable'}
                </span>
              </div>
            </div>

            {/* Card B: Rubberband */}
            <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-5 flex flex-col justify-between min-h-[140px]">
              <div className="flex items-center justify-between mb-4">
                <Cpu className="w-4 h-4 text-[#adc6ff]" />
                {health.dependencies.rubberband.available ? (
                  <span className="font-mono text-[9px] bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 text-emerald-400 rounded">
                    READY
                  </span>
                ) : (
                  <span className="font-mono text-[9px] bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 text-amber-400 rounded">
                    CLI MISSING
                  </span>
                )}
              </div>
              <div>
                <span className="text-[10px] font-mono text-[#c2c6d6] block mb-1">Time-Stretch</span>
                <span className="text-base font-bold text-[#e5e1e4] block">Rubberband</span>
                <span className="font-mono text-[10px] text-[#c2c6d6] opacity-80 truncate block">
                  {health.dependencies.rubberband.version ? `v${health.dependencies.rubberband.version.substring(0, 20)}` : 'FALLBACK mode'}
                </span>
              </div>
            </div>

            {/* Card C: GPU Details */}
            <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-5 flex flex-col justify-between min-h-[140px]">
              <div className="flex items-center justify-between mb-2">
                <Cpu className="w-4 h-4 text-[#adc6ff]" />
                <span className="font-mono text-[10px] bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 text-indigo-400 rounded uppercase">
                  {health.dependencies.gpu.available ? 'CUDA Active' : 'CPU Only'}
                </span>
              </div>
              <div>
                <span className="text-[10px] font-mono text-[#c2c6d6] block mb-1">
                  {health.dependencies.gpu.name ? health.dependencies.gpu.name.substring(0, 22) : 'No GPU Detected'}
                </span>
                <span className="text-base font-bold text-[#e5e1e4] block">
                  {health.dependencies.gpu.available
                    ? `${(health.dependencies.gpu.vram_mb / 1024).toFixed(1)} GB VRAM`
                    : 'System Memory fallback'}
                </span>
              </div>
            </div>

            {/* Card D: Active Job Capacity */}
            <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-5 flex flex-col justify-between min-h-[140px]">
              <div className="flex items-center justify-between mb-2">
                <HardDrive className="w-4 h-4 text-[#adc6ff]" />
                <span className="font-mono text-sm font-bold text-[#e5e1e4]">
                  {health.active_jobs} / {health.max_concurrent_jobs}
                </span>
              </div>
              <div>
                <div className="flex justify-between items-center text-[10px] font-mono text-[#c2c6d6] mb-1">
                  <span>Job Slots Active</span>
                  <span>{Math.round((health.active_jobs / health.max_concurrent_jobs) * 100)}%</span>
                </div>
                {/* Status illuminated dots indicator */}
                <div className="flex items-center gap-1.5 pt-1">
                  {Array.from({ length: health.max_concurrent_jobs }).map((_, idx) => (
                    <div
                      key={idx}
                      className={`h-2 flex-1 rounded-full ${idx < health.active_jobs
                          ? 'bg-blue-400 animate-pulse'
                          : 'bg-[#1c1b1d] border border-[#424754]/30'
                        }`}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

          </div>

          {/* Storage Volumes Block */}
          <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-6">
            <h3 className="font-sans font-bold text-sm tracking-tight text-[#e5e1e4] mb-6 flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-[#adc6ff]" />
              <span>Storage Volumes</span>
            </h3>

            <div className="flex flex-col gap-6">
              {/* Volume 1: Ingestion uploads */}
              <div>
                <div className="flex justify-between text-xs mb-1.5 font-mono">
                  <div>
                    <span className="text-[#e5e1e4] font-semibold">./data/uploads</span>
                    <span className="text-[#c2c6d6] block text-[10px] mt-0.5">Raw video & audio ingestion</span>
                  </div>
                  <span className="text-[#e5e1e4] font-medium">{uploadsSize} MB</span>
                </div>
                {/* Loader bar (representing size, using a 500MB scale for visualization) */}
                <div className="w-full bg-[#1c1b1d] h-2.5 rounded-full border border-[#1c1b1d] overflow-hidden">
                  <div
                    className="bg-indigo-500 h-full transition-all duration-1000"
                    style={{ width: `${Math.min(100, (uploadsSize / 500) * 100)}%` }}
                  ></div>
                </div>
              </div>

              {/* Volume 2: Rendered outputs */}
              <div>
                <div className="flex justify-between text-xs mb-1.5 font-mono">
                  <div>
                    <span className="text-[#e5e1e4] font-semibold">./data/outputs (./output)</span>
                    <span className="text-[#c2c6d6] block text-[10px] mt-0.5">Rendered audio stems & final muxed videos</span>
                  </div>
                  <span className="text-[#e5e1e4] font-medium">{outputsSize} MB</span>
                </div>
                {/* Loader bar (representing size, using a 1000MB scale for visualization) */}
                <div className="w-full bg-[#1c1b1d] h-2.5 rounded-full border border-[#1c1b1d] overflow-hidden">
                  <div
                    className="bg-amber-500 h-full transition-all duration-1000"
                    style={{ width: `${Math.min(100, (outputsSize / 1000) * 100)}%` }}
                  ></div>
                </div>
              </div>

            </div>
          </div>

        </div>
      ) : activeTab === 'config' ? (
        /* Configuration settings tab content */
        <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-6 max-w-3xl">
          <h3 className="font-sans font-bold text-sm tracking-tight text-[#e5e1e4] mb-6 flex items-center gap-2 pb-3 border-b border-[#1c1b1d]">
            <Sliders className="w-4 h-4 text-[#adc6ff]" />
            <span>Active Pipeline configurations</span>
          </h3>

          {!config ? (
            <div className="py-8 text-center text-xs text-[#c2c6d6] font-mono flex items-center justify-center gap-2">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Loading engine configuration files...</span>
            </div>
          ) : (
            <div className="flex flex-col gap-6 font-mono text-xs">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Panel 1: VLM Client */}
                <div className="bg-[#0d0d0f] rounded-lg p-4 border border-[#1c1b1d]">
                  <h4 className="font-sans font-bold text-[#adc6ff] mb-3 text-xs uppercase tracking-wider">Vision-Language Model</h4>
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">VLM Mode:</span><b className="text-[#e5e1e4]">{config.vlm.mode}</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">Model Reference:</span><b className="text-[#e5e1e4] truncate max-w-[150px]" title={config.vlm.model_name}>{config.vlm.model_name}</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">VLM Temperature:</span><b className="text-[#e5e1e4]">{config.vlm.temperature}</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">Max Output Tokens:</span><b className="text-[#e5e1e4]">{config.vlm.max_tokens}</b></div>
                  </div>
                </div>

                {/* Panel 2: TTS Engine */}
                <div className="bg-[#0d0d0f] rounded-lg p-4 border border-[#1c1b1d]">
                  <h4 className="font-sans font-bold text-[#adc6ff] mb-3 text-xs uppercase tracking-wider">Zero-Shot Voice Cloning</h4>
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">TTS Engine:</span><b className="text-[#e5e1e4]">{config.tts.engine}</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">Sample Output Rate:</span><b className="text-[#e5e1e4]">{config.tts.sample_rate} Hz</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">VieNeu Synthesizer:</span><b className="text-[#e5e1e4]">{config.tts.vieneu_mode}</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">Backbone CUDA Dev:</span><b className="text-[#e5e1e4]">{config.tts.backbone_device}</b></div>
                  </div>
                </div>

                {/* Panel 3: Pipeline Flow */}
                <div className="bg-[#0d0d0f] rounded-lg p-4 border border-[#1c1b1d]">
                  <h4 className="font-sans font-bold text-[#adc6ff] mb-3 text-xs uppercase tracking-wider">Pipeline Pipeline settings</h4>
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">Default Mode:</span><b className="text-[#e5e1e4] uppercase">{config.pipeline.mode}</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">VLM Scene Detection:</span><b className="text-[#e5e1e4]">{config.pipeline.chunk_duration_sec}s chunks</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">Scene Threshold:</span><b className="text-[#e5e1e4]">{config.pipeline.scene_threshold}</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">SSIM Dedup Ratio:</span><b className="text-[#e5e1e4]">{config.pipeline.ssim_threshold}</b></div>
                  </div>
                </div>

                {/* Panel 4: Audio Synchronizer & Limits */}
                <div className="bg-[#0d0d0f] rounded-lg p-4 border border-[#1c1b1d]">
                  <h4 className="font-sans font-bold text-[#adc6ff] mb-3 text-xs uppercase tracking-wider">Sync & Upload limits</h4>
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">Sync Max Speedup:</span><b className="text-[#e5e1e4]">{config.sync.max_speedup}x (Rubberband)</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">Silence Gap Min:</span><b className="text-[#e5e1e4]">{config.sync.min_gap_sec}s</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">Max Video size:</span><b className="text-[#e5e1e4]">{config.limits.max_video_size_mb} MB</b></div>
                    <div className="flex justify-between"><span className="text-[#c2c6d6]">Max Audio Voice size:</span><b className="text-[#e5e1e4]">{config.limits.max_audio_size_mb} MB</b></div>
                  </div>
                </div>
              </div>

              <div className="bg-[#1c1b1d] rounded-lg p-4 border border-[#1c1b1d] mt-2 flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-[#adc6ff] shrink-0 mt-0.5" />
                <div className="font-sans">
                  <p className="text-xs font-semibold text-[#e5e1e4]">Local Storage Settings applied</p>
                  <p className="text-[11px] text-[#c2c6d6] mt-0.5 leading-normal">
                    These settings represent configurations stored in `configs/pipeline_config.json` and system env parameters. Sensitive API keys are hidden.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
