import React, { useState, useRef, useEffect } from 'react';
import { Video, HelpCircle, FileAudio, ToggleLeft, ToggleRight, Play, CheckCircle2, ChevronDown, RefreshCw, Sparkles, AlertTriangle, Users } from 'lucide-react';
import { api, VoiceOption } from '../api';

interface CreateNewDubProps {
  onAddJob?: (job: any) => void; // Kept for interface compatibility
  onNavigateToJobs: () => void;
  onJobCreated?: (jobId: string) => void;
}

const PRESET_VIDEOS = [
  { name: 'Demo-Module-5.mp4', path: 'data/video/Demo-Module-5.mp4', size: '3.3 MB', icon: '🧠' }
];

const PRESET_AUDIOS = [
  { name: 'speaker.wav', path: 'data/reference_audio/speaker.wav', desc: 'Preloaded English Speaker' }
];

export default function CreateNewDub({ onNavigateToJobs, onJobCreated }: CreateNewDubProps) {
  // Paths sent to the API
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [audioPath, setAudioPath] = useState<string | null>(null);

  // Selected Visual States
  const [selectedVideo, setSelectedVideo] = useState<{ name: string; size: string } | null>(null);
  const [selectedAudioName, setSelectedAudioName] = useState<string | null>(null);
  const [voiceSampleType, setVoiceSampleType] = useState<'upload' | 'preset'>('upload');

  // Preset voices
  const [presetVoices, setPresetVoices] = useState<VoiceOption[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null);

  // Upload progress and loading states
  const [uploadingVideo, setUploadingVideo] = useState(false);
  const [uploadingAudio, setUploadingAudio] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Drag-and-drop state
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  // Form inputs
  const [pipelineMode, setPipelineMode] = useState<'ocr' | 'vlm'>('ocr');
  const [vlmMode, setVlmMode] = useState<'local' | 'api'>('local');
  const [ttsEngine, setTtsEngine] = useState('vieneu');
  const [keepOriginalAudio, setKeepOriginalAudio] = useState(false);
  const [sourceLang, setSourceLang] = useState('English');
  const [targetLang, setTargetLang] = useState('Vietnamese');

  const [showTtsDropdown, setShowTtsDropdown] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  // Preset Selection Handlers
  const handleSelectPresetVideo = (preset: typeof PRESET_VIDEOS[0]) => {
    setVideoPath(preset.path);
    setSelectedVideo({ name: preset.name, size: preset.size });
    setSubmitError(null);
  };

  const handleSelectPresetAudio = (preset: typeof PRESET_AUDIOS[0]) => {
    setAudioPath(preset.path);
    setSelectedAudioName(preset.name);
    setSubmitError(null);
  };

  // Video Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      await handleVideoUpload(file);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      await handleVideoUpload(file);
    }
  };

  const handleVideoUpload = async (file: File) => {
    setUploadingVideo(true);
    setSubmitError(null);
    try {
      const res = await api.uploadVideo(file);
      setVideoPath(res.path);
      setSelectedVideo({
        name: file.name,
        size: `${res.size_mb.toFixed(1)} MB`
      });
    } catch (err: any) {
      console.error(err);
      setSubmitError(`Video upload failed: ${err.message}`);
    } finally {
      setUploadingVideo(false);
    }
  };

  // Audio Upload Handlers
  const handleAudioFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      await handleAudioUpload(file);
    }
  };

  const handleAudioUpload = async (file: File) => {
    setUploadingAudio(true);
    setSubmitError(null);
    try {
      const res = await api.uploadAudio(file);
      setAudioPath(res.path);
      setSelectedAudioName(file.name);
    } catch (err: any) {
      console.error(err);
      setSubmitError(`Audio upload failed: ${err.message}`);
    } finally {
      setUploadingAudio(false);
    }
  };

  // Fetch preset voices once on mount
  useEffect(() => {
    api.listVoices()
      .then((res) => {
        setPresetVoices(res.voices);
        if (res.voices.length > 0) {
          setSelectedPresetId(res.default ?? res.voices[0].id);
        }
      })
      .catch((err) => console.error('Failed to load preset voices', err));
  }, []);

  // Build and submit the dubbing job
  const handleGenerate = async () => {
    if (!videoPath) {
      setSubmitError('Please select or upload a source video.');
      return;
    }

    const usingPreset = voiceSampleType === 'preset';
    if (usingPreset) {
      if (!selectedPresetId) {
        setSubmitError('Please choose a VieNeu preset voice.');
        return;
      }
    } else if (!audioPath) {
      setSubmitError('Please select or upload a voice reference audio.');
      return;
    }

    setIsGenerating(true);
    setSubmitError(null);

    try {
      const res = await api.startProcess({
        video_path: videoPath,
        audio_path: usingPreset ? undefined : audioPath!,
        voice_source: usingPreset ? 'preset' : 'clone',
        preset_voice_id: usingPreset ? selectedPresetId! : undefined,
        pipeline_mode: pipelineMode,
        vlm_mode: vlmMode,
        tts_engine: ttsEngine,
        source_lang: sourceLang,
        target_lang: targetLang,
        keep_original_audio: keepOriginalAudio,
      });
      setIsGenerating(false);
      if (onJobCreated) {
        onJobCreated(res.job_id);
      } else {
        onNavigateToJobs();
      }
    } catch (err: any) {
      console.error(err);
      setSubmitError(`Failed to start dubbing job: ${err.message}`);
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col min-h-screen bg-[#0d0d0f] text-[#e5e1e4] p-8 pb-16">
      {/* Top Breadcrumb */}
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs text-[#c2c6d6] hover:underline cursor-pointer" onClick={onNavigateToJobs}>
          Jobs
        </span>
        <span className="text-[#424754] text-xs">/</span>
        <span className="text-xs text-[#adc6ff] font-medium">New Dub</span>
      </div>

      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="font-sans font-bold text-3xl tracking-tight text-[#e5e1e4] mb-2">Create New Dub</h1>
          <p className="text-sm text-[#c2c6d6]">Upload your source media and configure the AI voice engine pipeline.</p>
        </div>
        <div className="bg-[#1c1b1d] border border-[#424754] px-3 py-1.5 rounded-md flex items-center gap-2">
          <div className="w-2 h-2 bg-emerald-500 rounded-full animate-ping"></div>
          <span className="font-mono text-[10px] text-[#adc6ff] tracking-widest uppercase">ENGINE READY</span>
        </div>
      </div>

      {submitError && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div className="text-xs text-red-400">
            <span className="font-semibold block mb-0.5">Configuration Error</span>
            <span>{submitError}</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 items-start">
        {/* Left Column - Inputs (8 columns on wide screen) */}
        <div className="xl:col-span-8 flex flex-col gap-6">

          {/* Module 1: Source Video */}
          <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-6">
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-[#1c1b1d]">
              <div className="flex items-center gap-2.5">
                <Video className="w-4 h-4 text-[#adc6ff]" />
                <span className="text-sm font-semibold text-[#e5e1e4]">Source Video</span>
              </div>
              <span className="font-mono text-[10px] text-[#c2c6d6]">Max 200MB • MP4, MKV, WEBM</span>
            </div>

            {/* Drag and Drop Container */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => !uploadingVideo && fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-8 flex flex-col items-center justify-center text-center transition-all cursor-pointer ${isDragging
                ? 'border-[#adc6ff] bg-[#adc6ff]/5'
                : selectedVideo
                  ? 'border-emerald-500/50 bg-emerald-500/5'
                  : 'border-[#1c1b1d] hover:border-[#adc6ff]/30 hover:bg-[#1c1b1d]/40'
                } ${uploadingVideo ? 'pointer-events-none opacity-60' : ''}`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4, video/x-matroska, video/webm, video/quicktime"
                onChange={handleFileSelect}
                className="hidden"
              />

              {uploadingVideo ? (
                <div className="flex flex-col items-center">
                  <RefreshCw className="w-8 h-8 animate-spin text-[#adc6ff] mb-3" />
                  <p className="text-sm text-[#e5e1e4] font-medium">Uploading video to server...</p>
                  <p className="text-xs text-[#c2c6d6] mt-1">Please keep this window open</p>
                </div>
              ) : selectedVideo ? (
                <div className="flex flex-col items-center">
                  <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 mb-3 animate-bounce">
                    <CheckCircle2 className="w-6 h-6" />
                  </div>
                  <p className="text-sm font-medium text-emerald-400 mb-1">{selectedVideo.name}</p>
                  <p className="font-mono text-[11px] text-[#c2c6d6] mb-4">{selectedVideo.size}</p>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedVideo(null);
                      setVideoPath(null);
                    }}
                    className="text-xs text-red-400 hover:underline cursor-pointer"
                  >
                    Remove and choose another
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <div className="w-12 h-12 rounded-lg bg-[#1c1b1d] flex items-center justify-center text-gray-400 mb-4 border border-[#424754]">
                    <svg className="w-5 h-5 text-[#adc6ff]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                  </div>
                  <p className="text-sm text-[#e5e1e4] font-medium mb-1">Drag and drop your video file here</p>
                  <p className="text-xs text-[#c2c6d6] mb-4">or click to browse from your computer</p>
                  <span className="px-4 py-1.5 rounded-md bg-[#2a2a2c] text-xs font-semibold hover:bg-[#353437] transition border border-[#424754]">
                    Select File
                  </span>
                </div>
              )}
            </div>

            {/* Presets Grid */}
            <div className="mt-4 pt-4 border-t border-[#1c1b1d]">
              <p className="text-xs text-[#c2c6d6] mb-2 font-medium">Or quickly use a local demo template:</p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                {PRESET_VIDEOS.map((preset) => (
                  <button
                    key={preset.name}
                    type="button"
                    onClick={() => handleSelectPresetVideo(preset)}
                    className={`p-2.5 rounded-md border text-left flex flex-col justify-between transition cursor-pointer ${videoPath === preset.path
                      ? 'bg-emerald-500/10 border-emerald-500 text-[#e5e1e4]'
                      : 'bg-[#1c1b1d] border-[#1c1b1d] hover:border-[#424754] text-[#c2c6d6] hover:text-[#e5e1e4]'
                      }`}
                  >
                    <span className="text-lg mb-1">{preset.icon}</span>
                    <span className="text-[11px] font-semibold truncate block w-full">{preset.name}</span>
                    <span className="font-mono text-[9px] text-[#c2c6d6]/80">{preset.size} (Preloaded)</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Module 2: Target Voice Sample */}
          <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-6">
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-[#1c1b1d]">
              <div className="flex items-center gap-2.5">
                <FileAudio className="w-4 h-4 text-[#adc6ff]" />
                <span className="text-sm font-semibold text-[#e5e1e4]">Target Voice Sample</span>
              </div>
              <span className="font-mono text-[10px] text-[#c2c6d6]">Max 10MB • WAV, MP3</span>
            </div>

            {/* Selector Option tabs */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <button
                type="button"
                onClick={() => setVoiceSampleType('upload')}
                className={`p-4 rounded-lg border text-center flex flex-col items-center justify-center gap-2 transition cursor-pointer ${voiceSampleType === 'upload'
                  ? 'bg-[#2a2a2c]/65 border-[#adc6ff] text-[#adc6ff]'
                  : 'bg-[#1c1b1d] border-[#1c1b1d] hover:border-[#424754] text-[#c2c6d6] hover:text-[#e5e1e4]'
                  }`}
              >
                <div className="w-8 h-8 rounded-full bg-[#0d0d0f] flex items-center justify-center border border-[#1c1b1d]">
                  <FileAudio className="w-4 h-4" />
                </div>
                <div>
                  <span className="text-xs font-bold block">Upload / Preset File</span>
                  <span className="font-mono text-[9px] text-[#c2c6d6]">Clone a voice (WAV, MP3, M4A)</span>
                </div>
              </button>

              <button
                type="button"
                onClick={() => setVoiceSampleType('preset')}
                className={`p-4 rounded-lg border text-center flex flex-col items-center justify-center gap-2 transition cursor-pointer ${voiceSampleType === 'preset'
                  ? 'bg-[#2a2a2c]/65 border-[#adc6ff] text-[#adc6ff]'
                  : 'bg-[#1c1b1d] border-[#1c1b1d] hover:border-[#424754] text-[#c2c6d6] hover:text-[#e5e1e4]'
                  }`}
              >
                <div className="w-8 h-8 rounded-full bg-[#0d0d0f] flex items-center justify-center border border-[#1c1b1d]">
                  <Users className="w-4 h-4" />
                </div>
                <div>
                  <span className="text-xs font-bold block">VieNeu Voice</span>
                  <span className="font-mono text-[9px] text-[#c2c6d6]">Built-in narrator</span>
                </div>
              </button>
            </div>

            {/* Preset voice panel */}
            {voiceSampleType === 'preset' && (
              <div className="bg-[#1c1b1d] border border-[#1c1b1d] rounded-md p-4 mb-3">
                <span className="text-xs font-bold text-[#c2c6d6] block mb-2">Choose a built-in VieNeu narrator:</span>
                {presetVoices.length === 0 ? (
                  <p className="text-xs text-[#c2c6d6]">
                    No preset voices found in the local cache. Please use the Upload / Preset File tab instead.
                  </p>
                ) : (
                  <select
                    value={selectedPresetId ?? ''}
                    onChange={(e) => setSelectedPresetId(e.target.value)}
                    className="w-full bg-[#141416] border border-[#424754] text-xs px-2.5 py-2 rounded focus:outline-none focus:border-[#adc6ff]"
                  >
                    {presetVoices.map((v) => (
                      <option key={v.id} value={v.id}>{v.label}</option>
                    ))}
                  </select>
                )}
                <p className="text-[10px] text-[#c2c6d6] mt-2">Narration uses this generic voice (the original speaker is not cloned).</p>
              </div>
            )}

            {/* Upload Area (file upload + preloaded audio presets) */}
            {voiceSampleType !== 'preset' && (
              <div className="bg-[#1c1b1d] border border-[#1c1b1d] rounded-md p-4">
                <span className="text-xs font-bold text-[#c2c6d6] block mb-2">Select preloaded voice preset or upload file:</span>

                <div className="flex flex-wrap gap-2 mb-3">
                  {PRESET_AUDIOS.map((preset) => (
                    <button
                      key={preset.name}
                      type="button"
                      onClick={() => handleSelectPresetAudio(preset)}
                      className={`px-3 py-1.5 rounded-md text-xs font-mono transition border cursor-pointer ${audioPath === preset.path
                        ? 'bg-emerald-500/10 border-emerald-500 text-emerald-400'
                        : 'bg-[#141416] border-[#1c1b1d] hover:border-[#424754] text-[#c2c6d6]'
                        }`}
                    >
                      {preset.name}
                    </button>
                  ))}
                </div>

                <div className="flex items-center gap-3">
                  <input
                    ref={audioInputRef}
                    type="file"
                    accept="audio/wav, audio/mpeg, audio/flac, audio/ogg, audio/x-m4a"
                    onChange={handleAudioFileSelect}
                    className="hidden"
                  />
                  <button
                    type="button"
                    disabled={uploadingAudio}
                    onClick={() => audioInputRef.current?.click()}
                    className="px-3 py-1.5 bg-[#2a2a2c] hover:bg-[#353437] text-xs rounded border border-[#424754] text-[#e5e1e4] flex items-center gap-2 cursor-pointer disabled:opacity-60"
                  >
                    {uploadingAudio ? (
                      <RefreshCw className="w-3 h-3 animate-spin text-[#adc6ff]" />
                    ) : (
                      <FileAudio className="w-3 h-3 text-[#adc6ff]" />
                    )}
                    <span>Upload Custom Speaker reference</span>
                  </button>
                </div>

                {selectedAudioName && (
                  <div className="mt-3 flex items-center gap-2 text-xs text-emerald-400">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Loaded profile target: <b>{selectedAudioName}</b> {audioPath?.includes('uploads') ? '(Custom Uploaded)' : '(System Preset)'}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column - Pipeline configuration */}
        <div className="xl:col-span-4 flex flex-col gap-6">
          <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-6 flex flex-col h-full justify-between">
            <div>
              {/* Header */}
              <div className="flex items-center gap-2 mb-4 pb-3 border-b border-[#1c1b1d]">
                <Sparkles className="w-4 h-4 text-[#adc6ff]" />
                <div className="flex flex-col">
                  <span className="text-sm font-semibold text-[#e5e1e4]">Pipeline Configuration</span>
                  <span className="text-[10px] text-[#c2c6d6]">Advanced settings for generation</span>
                </div>
              </div>

              {/* Languages Selection */}
              <div className="mb-5">
                <label className="font-mono text-[10px] text-[#adc6ff] tracking-wider uppercase block mb-2">
                  Language settings
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-[10px] text-[#c2c6d6] block mb-1">Source (Original)</span>
                    <select
                      value={sourceLang}
                      onChange={(e) => setSourceLang(e.target.value)}
                      className="w-full bg-[#1c1b1d] border border-[#1c1b1d] text-xs px-2.5 py-1.5 rounded focus:outline-none focus:border-[#adc6ff]"
                    >
                      <option value="English">English</option>
                      <option value="Vietnamese">Vietnamese</option>
                      <option value="French">French</option>
                      <option value="Chinese">Chinese</option>
                    </select>
                  </div>
                  <div>
                    <span className="text-[10px] text-[#c2c6d6] block mb-1">Target (Dubbed)</span>
                    <select
                      value={targetLang}
                      onChange={(e) => setTargetLang(e.target.value)}
                      className="w-full bg-[#1c1b1d] border border-[#1c1b1d] text-xs px-2.5 py-1.5 rounded focus:outline-none focus:border-[#adc6ff]"
                    >
                      <option value="Vietnamese">Vietnamese</option>
                      <option value="English">English</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Extraction Engine */}
              <div className="mb-5">
                <label className="font-mono text-[10px] text-[#adc6ff] tracking-wider uppercase block mb-2">
                  Extraction Pipeline Mode
                </label>
                <div className="grid grid-cols-2 rounded bg-[#0d0d0f] p-1 border border-[#1c1b1d]">
                  <button
                    type="button"
                    onClick={() => setPipelineMode('vlm')}
                    className={`py-1.5 text-xs font-semibold rounded cursor-pointer transition ${pipelineMode === 'vlm'
                      ? 'bg-[#1c1b1d] text-[#e5e1e4] shadow-sm'
                      : 'text-[#c2c6d6] hover:text-[#e5e1e4]'
                      }`}
                  >
                    VLM (Vision-Language)
                  </button>
                  <button
                    type="button"
                    onClick={() => setPipelineMode('ocr')}
                    className={`py-1.5 text-xs font-semibold rounded cursor-pointer transition ${pipelineMode === 'ocr'
                      ? 'bg-[#1c1b1d] text-[#e5e1e4] shadow-sm'
                      : 'text-[#c2c6d6] hover:text-[#e5e1e4]'
                      }`}
                  >
                    OCR + VLM
                  </button>
                </div>
                <span className="text-[9px] text-[#c2c6d6] block mt-1">
                  OCR-driven mode extracts burned-in subtitles. VLM mode splits by scenes.
                </span>
              </div>

              {/* VLM Mode (local vs api) */}
              <div className="mb-5">
                <label className="font-mono text-[10px] text-[#adc6ff] tracking-wider uppercase block mb-2">
                  VLM Translation Mode
                </label>
                <div className="grid grid-cols-2 rounded bg-[#0d0d0f] p-1 border border-[#1c1b1d]">
                  <button
                    type="button"
                    onClick={() => setVlmMode('local')}
                    className={`py-1.5 text-xs font-semibold rounded cursor-pointer transition ${vlmMode === 'local'
                      ? 'bg-[#1c1b1d] text-[#e5e1e4] shadow-sm'
                      : 'text-[#c2c6d6] hover:text-[#e5e1e4]'
                      }`}
                  >
                    Local Qwen3.5-2B
                  </button>
                  <button
                    type="button"
                    onClick={() => setVlmMode('api')}
                    className={`py-1.5 text-xs font-semibold rounded cursor-pointer transition ${vlmMode === 'api'
                      ? 'bg-[#1c1b1d] text-[#e5e1e4] shadow-sm'
                      : 'text-[#c2c6d6] hover:text-[#e5e1e4]'
                      }`}
                  >
                    Gemini API Key
                  </button>
                </div>
              </div>

              {/* TTS Synthesis Model */}
              <div className="mb-6 relative">
                <label className="font-mono text-[10px] text-[#adc6ff] tracking-wider uppercase block mb-2">
                  TTS Engine
                </label>
                <button
                  type="button"
                  onClick={() => setShowTtsDropdown(!showTtsDropdown)}
                  className="w-full bg-[#1c1b1d] border border-[#1c1b1d] rounded px-3 py-2 flex items-center justify-between text-xs text-left cursor-pointer hover:border-[#424754] transition font-mono"
                >
                  <span>{ttsEngine === 'vieneu' ? 'VieNeu-TTS (Turbo v2)' : ttsEngine}</span>
                  <ChevronDown className="w-3.5 h-3.5 text-[#c2c6d6]" />
                </button>

                {showTtsDropdown && (
                  <div className="absolute top-full left-0 w-full bg-[#1c1b1d] border border-[#424754] rounded-md mt-1 shadow-xl z-10 p-1 flex flex-col gap-0.5">
                    <button
                      type="button"
                      onClick={() => {
                        setTtsEngine('vieneu');
                        setShowTtsDropdown(false);
                      }}
                      className="w-full text-left font-mono px-2.5 py-1.5 rounded hover:bg-[#2a2a2c] text-xs text-[#adc6ff]"
                    >
                      VieNeu-TTS (Turbo v2)
                    </button>
                  </div>
                )}
              </div>

              {/* Toggle Option: Keep original audio */}
              <div className="mb-6 pt-4 border-t border-[#1c1b1d]">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-xs font-semibold block text-[#e5e1e4]">Keep Original Audio</span>
                    <span className="text-[10px] text-[#c2c6d6] max-w-[210px] block mt-0.5 leading-normal">
                      Background audio mixed at -18dB volume.
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setKeepOriginalAudio(!keepOriginalAudio)}
                    className="text-[#adc6ff] hover:text-[#adc6ff]/90 cursor-pointer"
                  >
                    {keepOriginalAudio ? (
                      <ToggleRight className="w-9 h-9" />
                    ) : (
                      <ToggleLeft className="w-9 h-9 text-[#424754]" />
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Action buttons */}
            <div className="mt-8 border-t border-[#1c1b1d] pt-6">
              <button
                type="button"
                onClick={handleGenerate}
                disabled={isGenerating || uploadingVideo || uploadingAudio}
                className="w-full py-3.5 px-4 rounded bg-[#3b82f6] hover:bg-[#3b82f6]/95 text-white font-medium text-sm flex items-center justify-center gap-2 transition-all cursor-pointer shadow-lg shadow-[#3b82f6]/20 disabled:opacity-60 disabled:pointer-events-none"
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-white" />
                    <span>Allocating CPU/GPU block...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-white text-white" />
                    <span>Generate Dubbed Video</span>
                  </>
                )}
              </button>

              <div className="text-center mt-2">
                <span className="font-mono text-[10px] text-[#c2c6d6]">
                  Estimated speed: <b className="text-[#adc6ff]">~2.5x</b> real video length
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
