export interface PipelineStep {
  name: string;
  description: string;
  status: 'pending' | 'active' | 'completed';
  duration?: string;
  progress?: number; // for current active step
}

export interface Job {
  id: string;
  sourceMedia: string;
  sourceLang: string;
  targetLangs: string[];
  status: 'PROCESSING' | 'COMPLETE' | 'FAILED' | 'QUEUED';
  overallProgress: number;
  currentStep: string;
  createdText: string;
  createdAt: number;
  sourceSize: string;
  pipelineMode: 'VLM' | 'OCR';
  ttsModel: string;
  keepOriginalAudio: boolean;
  targetVoiceSampleName: string;
  targetVoiceSampleType: 'upload' | 'mic';
  logs: string[];
  steps: PipelineStep[];
}

export interface SystemMetrics {
  uptime: string;
  gpuVramUsage: number; // in GB out of 24GB
  activeJobs: number;
  totalCapacity: number;
  uploadsUsed: number; // GB
  uploadsTotal: number; // GB
  outputsUsed: number; // GB
  outputsTotal: number; // GB
  ffmpegVersion: string;
  rubberbandVersion: string;
  activeTab: 'metrics' | 'config';
}
