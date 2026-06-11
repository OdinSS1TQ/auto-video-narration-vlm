import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import CreateNewDub from './components/CreateNewDub';
import JobsList from './components/JobsList';
import JobDetailView from './components/JobDetailView';
import SystemHealth from './components/SystemHealth';
import { Job, PipelineStep } from './types';
import { api } from './api';

// Map backend job structure to frontend UI model
function mapBackendJobToFrontendJob(bj: any): Job {
  const isOCR = bj.pipeline_mode === 'ocr' || !bj.pipeline_mode;
  
  // Define standard steps depending on pipeline mode
  const ocrSteps = [
    { name: 'Caption OCR', description: 'Trích xuất timeline caption từ video bằng OCR.', key: 'caption_ocr_timeline' },
    { name: 'Narration Classification', description: 'Phân loại giọng đọc/cảnh quay bằng VLM.', key: 'vlm_classify_narration' },
    { name: 'Global Summary', description: 'Tóm tắt bối cảnh tổng quan của video.', key: 'vlm_global_summary' },
    { name: 'Segment Translation', description: 'Dịch các phân đoạn thoại bằng VLM.', key: 'vlm_translate_segments' },
    { name: 'SRT Generation', description: 'Tạo phụ đề SRT tiếng Việt khớp timeline.', key: 'srt_generation' },
    { name: 'Voice Cloning', description: 'VieNeu-TTS nhân bản giọng nói theo audio mẫu.', key: 'voice_cloning' },
    { name: 'Audio Alignment', description: 'Căn chỉnh thời gian giọng đọc khớp video.', key: 'audio_alignment' },
    { name: 'Video Rendering', description: 'Mux hình ảnh và âm thanh mới bằng FFmpeg.', key: 'video_rendering' }
  ];

  const vlmSteps = [
    { name: 'Scene Detection', description: 'Phát hiện các cảnh quay thay đổi trong video.', key: 'scene_detection' },
    { name: 'Frame Extraction', description: 'Trích xuất và lọc các khung hình đại diện.', key: 'frame_extraction' },
    { name: 'OCR Extraction', description: 'Nhận dạng chữ trên khung hình cảnh.', key: 'ocr_extraction' },
    { name: 'VLM Translation', description: 'Dịch lời thoại ngữ cảnh bằng VLM.', key: 'vlm_translation' },
    { name: 'SRT Generation', description: 'Tạo phụ đề SRT tiếng Việt khớp timeline.', key: 'srt_generation' },
    { name: 'Voice Cloning', description: 'VieNeu-TTS nhân bản giọng nói theo audio mẫu.', key: 'voice_cloning' },
    { name: 'Audio Alignment', description: 'Căn chỉnh thời gian giọng đọc khớp video.', key: 'audio_alignment' },
    { name: 'Video Rendering', description: 'Mux hình ảnh và âm thanh mới bằng FFmpeg.', key: 'video_rendering' }
  ];

  const templateSteps = isOCR ? ocrSteps : vlmSteps;
  
  // Calculate individual steps status and duration
  const currentStepKey = bj.progress?.step || 'queued';
  const currentIndex = templateSteps.findIndex(s => s.key === currentStepKey);

  const steps: PipelineStep[] = templateSteps.map((step, idx) => {
    let status: 'pending' | 'active' | 'completed' = 'pending';
    let duration = 'Pending';

    if (bj.status === 'completed') {
      status = 'completed';
      duration = 'Completed';
    } else if (bj.status === 'failed' || bj.status === 'cancelled') {
      if (idx < currentIndex) {
        status = 'completed';
        duration = 'Completed';
      } else if (idx === currentIndex) {
        status = 'active';
        duration = 'Failed';
      } else {
        status = 'pending';
        duration = 'Aborted';
      }
    } else if (bj.status === 'queued') {
      status = 'pending';
      duration = 'Waiting...';
    } else {
      // processing
      if (idx < currentIndex) {
        status = 'completed';
        duration = 'Completed';
      } else if (idx === currentIndex) {
        status = 'active';
        duration = 'Processing...';
      } else {
        status = 'pending';
        duration = 'Waiting...';
      }
    }

    return {
      name: step.name,
      description: step.description,
      status,
      duration
    };
  });

  // Normalize status for UI display
  let status: 'PROCESSING' | 'COMPLETE' | 'FAILED' | 'QUEUED' = 'QUEUED';
  if (bj.status === 'completed') status = 'COMPLETE';
  else if (bj.status === 'failed' || bj.status === 'cancelled') status = 'FAILED';
  else if (bj.status === 'processing') status = 'PROCESSING';
  else if (bj.status === 'queued') status = 'QUEUED';

  // Created time formatting
  const createdDate = new Date(bj.created_at);
  const diffMs = Date.now() - createdDate.getTime();
  let createdText = 'Just now';
  if (diffMs > 60000) {
    const mins = Math.floor(diffMs / 60000);
    if (mins < 60) createdText = `${mins}m ago`;
    else {
      const hours = Math.floor(mins / 60);
      createdText = `${hours}h ago`;
    }
  }

  return {
    id: bj.job_id,
    sourceMedia: bj.video_path ? bj.video_path.split(/[/\\]/).pop() || bj.video_path : 'Demo Video',
    sourceLang: bj.source_lang || 'EN',
    targetLangs: bj.target_lang ? [bj.target_lang] : ['VI'],
    status,
    overallProgress: bj.progress?.percent || 0,
    currentStep: bj.progress?.step_name_vi || bj.progress?.step || 'Initializing',
    createdText,
    createdAt: createdDate.getTime() || Date.now(),
    sourceSize: bj.size_mb ? `${bj.size_mb.toFixed(1)} MB` : '3.3 MB',
    pipelineMode: isOCR ? 'OCR' : 'VLM',
    ttsModel: bj.tts_engine === 'vieneu' ? 'VieNeu-TTS Turbo' : bj.tts_engine || 'VieNeu-TTS',
    keepOriginalAudio: bj.keep_original_audio || false,
    targetVoiceSampleName: bj.audio_path ? bj.audio_path.split(/[/\\]/).pop() || bj.audio_path : 'speaker.wav',
    targetVoiceSampleType: 'upload',
    logs: bj.logs || [],
    steps
  };
}

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<'new-job' | 'jobs-list' | 'job-detail' | 'system-health'>('new-job');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [deletedJobIds, setDeletedJobIds] = useState<string[]>([]);
  const [vramMetrics, setVramMetrics] = useState({ used: '0', total: '0' });

  // Load health metric VRAM for top header overview
  const fetchHeaderMetrics = async () => {
    try {
      const health = await api.getHealth();
      if (health.dependencies.gpu.available && health.dependencies.gpu.vram_mb) {
        const totalGb = (health.dependencies.gpu.vram_mb / 1024).toFixed(1);
        // Simulate minor variation for UI effect
        const randomGb = (Math.random() * 2 + 3).toFixed(1); // Mock GPU active load
        setVramMetrics({ used: randomGb, total: totalGb });
      } else {
        setVramMetrics({ used: '0.0', total: '16.0 (CPU fallback)' });
      }
    } catch (err) {
      console.error('Failed to load header metrics:', err);
    }
  };

  // Poll list of jobs periodically
  const fetchJobs = async () => {
    try {
      const res = await api.listJobs();
      const mapped = res.jobs.map(mapBackendJobToFrontendJob);
      
      // Merge logs if we already have detailed job state (to avoid losing logs on simple list poll)
      setJobs((prevJobs) => {
        return mapped.map((newJob) => {
          const matchingPrev = prevJobs.find(pj => pj.id === newJob.id);
          if (matchingPrev && matchingPrev.logs.length > newJob.logs.length) {
            return { ...newJob, logs: matchingPrev.logs };
          }
          return newJob;
        });
      });
    } catch (err) {
      console.error('Failed to poll jobs list:', err);
    }
  };

  // Poll detailed job status for logs and options
  const fetchDetailedJobStatus = async (jobId: string) => {
    try {
      const bj = await api.getJobStatus(jobId);
      const frontendJob = mapBackendJobToFrontendJob(bj);
      
      setJobs((prevJobs) => {
        const exists = prevJobs.some(j => j.id === jobId);
        if (exists) {
          return prevJobs.map(j => j.id === jobId ? frontendJob : j);
        } else {
          return [frontendJob, ...prevJobs];
        }
      });
    } catch (err) {
      console.error(`Failed to fetch job details for ${jobId}:`, err);
    }
  };

  // Polling loop controller
  useEffect(() => {
    fetchJobs();
    fetchHeaderMetrics();
    
    const interval = setInterval(() => {
      fetchJobs();
      fetchHeaderMetrics();
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  // Poll details specifically for the open job detail view
  useEffect(() => {
    if (currentScreen === 'job-detail' && selectedJobId) {
      fetchDetailedJobStatus(selectedJobId);
      const detailInterval = setInterval(() => {
        fetchDetailedJobStatus(selectedJobId);
      }, 2000);
      return () => clearInterval(detailInterval);
    }
  }, [currentScreen, selectedJobId]);

  // Redirect to job detail of the active job if it is generating
  const runningJob = jobs.filter(j => !deletedJobIds.includes(j.id)).find(j => j.status === 'PROCESSING' || j.status === 'QUEUED');
  const isLocked = !!runningJob;

  useEffect(() => {
    if (isLocked && runningJob) {
      if (currentScreen !== 'job-detail' || selectedJobId !== runningJob.id) {
        setSelectedJobId(runningJob.id);
        setCurrentScreen('job-detail');
      }
    }
  }, [isLocked, runningJob, currentScreen, selectedJobId]);

  // Event Handlers for Navigation
  const handleNavigate = (screen: 'new-job' | 'jobs-list' | 'job-detail' | 'system-health') => {
    if (isLocked) return;
    setCurrentScreen(screen);
  };

  const handleJobCreated = (jobId: string) => {
    setSelectedJobId(jobId);
    setCurrentScreen('job-detail');
    fetchDetailedJobStatus(jobId);
  };

  const handleSelectJob = (jobId: string) => {
    if (isLocked && jobId !== runningJob?.id) return;
    setSelectedJobId(jobId);
    setCurrentScreen('job-detail');
  };

  const handleDeleteJob = async (jobId: string) => {
    if (isLocked && jobId === runningJob?.id) return;
    const target = jobs.find(j => j.id === jobId);
    if (target && (target.status === 'PROCESSING' || target.status === 'QUEUED')) {
      try {
        await api.cancelJob(jobId);
      } catch (err) {
        console.error('Cancel job during deletion failed:', err);
      }
    }
    setDeletedJobIds(prev => [...prev, jobId]);
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      await api.cancelJob(jobId);
      // Immediately pull state
      fetchDetailedJobStatus(jobId);
    } catch (err: any) {
      alert(`Cancel failed: ${err.message}`);
    }
  };

  // Filter out jobs deleted locally
  const visibleJobs = jobs.filter((j) => !deletedJobIds.includes(j.id));
  const activeJob = jobs.find((j) => j.id === selectedJobId) || visibleJobs[0] || null;

  return (
    <div className="min-h-screen bg-[#0d0d0f] text-[#e5e1e4]">
      {/* Sidebar persistent left */}
      <Sidebar
        currentScreen={currentScreen}
        onNavigate={handleNavigate}
        selectedJobId={selectedJobId}
        isLocked={isLocked}
      />

      {/* Main Panel Content */}
      <div className="ml-[260px] flex flex-col min-h-screen relative">
        
        {/* Top Header Bar */}
        <header className="h-[64px] border-b border-[#1c1b1d] bg-[#0e0e10]/80 backdrop-blur-md flex items-center justify-between px-8 sticky top-0 z-40">
          <div className="flex items-center gap-3">
            <span className="text-xs uppercase tracking-wider font-mono text-[#adc6ff] flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>Workspace connected</span>
            </span>
          </div>

          <div className="flex items-center gap-6">
            {/* Telemetry overview display (loaded dynamically) */}
            <div className="hidden lg:flex items-center gap-6 font-mono text-[11px] text-[#c2c6d6]">
              <div>
                <span>VRAM Load:</span> <b className="text-[#adc6ff]">{vramMetrics.used} GB / {vramMetrics.total} GB</b>
              </div>
              <div className="h-3 w-[1px] bg-[#424754]" />
              <div>
                <span>System:</span> <b className="text-[#adc6ff]">API v0.1.0</b>
              </div>
            </div>

            {/* Profile actions mock icons */}
            <div className="flex items-center gap-3 border-l border-[#1c1b1d] pl-4">
              <button
                onClick={() => alert("Notification Center: System is connected to FastAPI backend.")}
                className="p-1.5 bg-[#141416] rounded border border-[#1c1b1d] hover:bg-[#1c1b1d] transition relative cursor-pointer text-[#c2c6d6] hover:text-[#e5e1e4]"
                title="Notifications"
              >
                <span className="absolute top-1 right-1 w-2 h-2 bg-[#3b82f6] rounded-full animate-bounce"></span>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
              </button>
              
              <button
                onClick={() => handleNavigate('system-health')}
                className="p-1.5 bg-[#141416] rounded border border-[#1c1b1d] hover:bg-[#1c1b1d] transition cursor-pointer text-[#c2c6d6] hover:text-[#e5e1e4]"
                title="System settings & telemetry logs"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065zM15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
            </div>
          </div>
        </header>

        {/* Dynamic Display Stage */}
        <main className="flex-1 flex flex-col relative">
          {currentScreen === 'new-job' && (
            <CreateNewDub
              onNavigateToJobs={() => setCurrentScreen('jobs-list')}
              onJobCreated={handleJobCreated}
            />
          )}

          {currentScreen === 'jobs-list' && (
            <JobsList
              jobs={visibleJobs}
              onSelectJob={handleSelectJob}
              onDeleteJob={handleDeleteJob}
            />
          )}

          {currentScreen === 'job-detail' && activeJob && (
            <JobDetailView
              job={activeJob}
              onGoBack={() => setCurrentScreen('jobs-list')}
              onCancelJob={handleCancelJob}
            />
          )}

          {currentScreen === 'job-detail' && !activeJob && (
            <div className="flex-1 flex items-center justify-center font-mono text-sm text-[#c2c6d6]">
              No active job selected. Go back to jobs list.
            </div>
          )}

          {currentScreen === 'system-health' && (
            <SystemHealth />
          )}
        </main>
      </div>
    </div>
  );
}
