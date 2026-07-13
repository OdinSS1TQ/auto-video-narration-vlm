/**
 * API client helper to interact with the FastAPI backend.
 */

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  timestamp: string;
  dependencies: {
    ffmpeg: { available: boolean; version?: string };
    rubberband: { available: boolean; version?: string };
    gpu: { available: boolean; name?: string; vram_mb?: number };
  };
  active_jobs: number;
  max_concurrent_jobs: number;
  disk_usage: {
    uploads_mb: number;
    outputs_mb: number;
  };
}

export interface ConfigResponse {
  vlm: {
    mode: string;
    model_name: string;
    temperature: number;
    max_tokens: number;
  };
  tts: {
    engine: string;
    sample_rate: number;
    vieneu_mode: string;
    backbone_device: string;
    codec_device: string;
  };
  pipeline: {
    mode: string;
    chunk_duration_sec: number;
    scene_threshold: number;
    ssim_threshold: number;
    max_concurrent_jobs: number;
  };
  sync: {
    max_speedup: number;
    min_gap_sec: number;
  };
  limits: {
    max_video_size_mb: number;
    max_audio_size_mb: number;
    max_video_duration_min: number;
    max_concurrent_jobs: number;
  };
  output: {
    dir: string;
    format: string;
  };
}

export interface UploadResponse {
  file_id: string;
  filename: string;
  path: string;
  size_mb: number;
}

export interface VoiceOption {
  id: string;
  label: string;
}

export interface VoiceListResponse {
  voices: VoiceOption[];
  default: string | null;
}

export interface ProcessResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface ProgressInfo {
  step: string;
  step_name_vi: string;
  current: number;
  total: number;
  percent: number;
}

export interface StatusResponse {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: ProgressInfo;
  created_at: string;
  error?: string | null;
  // Dynamic backend-extended fields
  logs?: string[];
  video_path?: string;
  audio_path?: string;
  pipeline_mode?: 'ocr' | 'vlm';
  vlm_mode?: string;
  keep_original_audio?: boolean;
  tts_engine?: string;
}

export interface JobSummary {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: ProgressInfo;
  created_at: string;
}

export interface JobListResponse {
  jobs: JobSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface ResultReportResponse {
  job_id: string;
  video_path: string;
  reference_audio: string;
  output_path: string;
  srt_path?: string | null;
  mode: string;
  elapsed_seconds: number;
  n_segments_raw?: number | null;
  n_segments_after_classify?: number | null;
  n_segments_merged?: number | null;
}

const BASE_API_URL = '/api';

/**
 * Handle HTTP errors gracefully
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = 'API call failed';
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || JSON.stringify(errJson);
    } catch {
      errorDetail = response.statusText || `${response.status} Error`;
    }
    throw new Error(errorDetail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  /**
   * Get system health metrics
   */
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${BASE_API_URL}/health`);
    return handleResponse<HealthResponse>(res);
  },

  /**
   * Get system pipeline configs
   */
  async getConfig(): Promise<ConfigResponse> {
    const res = await fetch(`${BASE_API_URL}/config`);
    return handleResponse<ConfigResponse>(res);
  },

  /**
   * Upload video file
   */
  async uploadVideo(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_API_URL}/upload/video`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<UploadResponse>(res);
  },

  /**
   * Upload reference audio file
   */
  async uploadAudio(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_API_URL}/upload/audio`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<UploadResponse>(res);
  },

  /**
   * List available VieNeu preset voices
   */
  async listVoices(): Promise<VoiceListResponse> {
    const res = await fetch(`${BASE_API_URL}/tts/voices`);
    return handleResponse<VoiceListResponse>(res);
  },

  /**
   * Start dubbing process pipeline
   */
  async startProcess(params: {
    video_path: string;
    audio_path?: string;
    voice_source?: 'clone' | 'preset';
    preset_voice_id?: string;
    vlm_mode?: string;
    tts_engine?: string;
    source_lang?: string;
    target_lang?: string;
    keep_original_audio?: boolean;
    pipeline_mode?: string;
  }): Promise<ProcessResponse> {
    const res = await fetch(`${BASE_API_URL}/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });
    return handleResponse<ProcessResponse>(res);
  },

  /**
   * Cancel queued or processing job
   */
  async cancelJob(jobId: string): Promise<ProcessResponse> {
    const res = await fetch(`${BASE_API_URL}/process/${jobId}`, {
      method: 'DELETE',
    });
    return handleResponse<ProcessResponse>(res);
  },

  /**
   * Query status and progress of a single job
   */
  async getJobStatus(jobId: string): Promise<StatusResponse> {
    const res = await fetch(`${BASE_API_URL}/status/${jobId}`);
    return handleResponse<StatusResponse>(res);
  },

  /**
   * List all jobs with filters and pagination
   */
  async listJobs(options?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<JobListResponse> {
    const params = new URLSearchParams();
    if (options?.status && options.status !== 'ALL') {
      params.append('status', options.status.toLowerCase());
    }
    if (options?.limit !== undefined) {
      params.append('limit', String(options.limit));
    }
    if (options?.offset !== undefined) {
      params.append('offset', String(options.offset));
    }

    const queryStr = params.toString() ? `?${params.toString()}` : '';
    const res = await fetch(`${BASE_API_URL}/jobs${queryStr}`);
    return handleResponse<JobListResponse>(res);
  },

  /**
   * Get metadata result report of a completed job
   */
  async getResultReport(jobId: string): Promise<ResultReportResponse> {
    const res = await fetch(`${BASE_API_URL}/result/${jobId}/report`);
    return handleResponse<ResultReportResponse>(res);
  },

  /**
   * Helper URLs for downloading final assets
   */
  getVideoDownloadUrl(jobId: string): string {
    return `${BASE_API_URL}/result/${jobId}`;
  },

  getSrtDownloadUrl(jobId: string): string {
    return `${BASE_API_URL}/result/${jobId}/srt`;
  }
};
