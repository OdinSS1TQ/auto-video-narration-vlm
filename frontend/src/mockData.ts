import { Job } from './types';

export const initialJobs: Job[] = [
  {
    id: 'JOB-8294',
    sourceMedia: 'Product_Launch_Q3_EN.mp4',
    sourceLang: 'EN',
    targetLangs: ['ES', 'FR', 'DE'],
    status: 'PROCESSING',
    overallProgress: 65,
    currentStep: 'Voice Synthesis',
    createdText: '12m ago',
    createdAt: Date.now() - 12 * 60 * 1000,
    sourceSize: '124 MB',
    pipelineMode: 'OCR',
    ttsModel: 'VieNeu-TTS Turbo (Recommended)',
    keepOriginalAudio: true,
    targetVoiceSampleName: 'vocal_reference_hq.wav',
    targetVoiceSampleType: 'upload',
    logs: [
      '[14:32:01] INF: Initializing pipeline for JOB-8294',
      '[14:32:02] INF: Extracting audio tracks... OK (2 channels found)',
      '[14:32:05] WRN: Low confidence in segment 12 (0.64), applying heuristic smoothing',
      '[14:32:45] INF: Translation model loaded (es-ES)',
      '[14:33:10] INF: Synthesizing voices with speaker profiles...',
      '[14:33:18] INF: Processing voice cloning chunk 14/35'
    ],
    steps: [
      { name: 'Caption OCR', description: 'Extracted 142 text segments from video frames.', status: 'completed', duration: '12s' },
      { name: 'Narration Classification', description: 'Identified 2 primary speakers and background noise.', status: 'completed', duration: '8s' },
      { name: 'Global Summary', description: 'Contextual analysis complete.', status: 'completed', duration: '15s' },
      { name: 'Segment Translation', description: 'Translated EN -> ES-ES. 100% confidence.', status: 'completed', duration: '42s' },
      { name: 'SRT Generation', description: 'Subtitles synchronized to original timing.', status: 'completed', duration: '5s' },
      { name: 'Voice Cloning', description: 'Synthesizing target audio using Speaker A & B models.', status: 'active', duration: 'Processing chunk 14/35...' },
      { name: 'Audio Alignment', description: 'Pending', status: 'pending' },
      { name: 'Video Rendering', description: 'Pending', status: 'pending' }
    ]
  },
  {
    id: 'JOB-8293',
    sourceMedia: 'CEO_Update_June.mov',
    sourceLang: 'EN',
    targetLangs: ['JA', 'KO'],
    status: 'COMPLETE',
    overallProgress: 100,
    currentStep: 'Ready for Download',
    createdText: '2h ago',
    createdAt: Date.now() - 120 * 60 * 1000,
    sourceSize: '82 MB',
    pipelineMode: 'VLM',
    ttsModel: 'VieNeu-TTS Turbo (Recommended)',
    keepOriginalAudio: true,
    targetVoiceSampleName: 'ceo_voice_recording.mp3',
    targetVoiceSampleType: 'upload',
    logs: [
      '[12:15:00] INF: Initializing pipeline for JOB-8293',
      '[12:15:02] INF: VLM engine initialized. Setting layout scanning resolution: 1080p',
      '[12:15:35] INF: Extracted speaker timeline (1 main speaker found)',
      '[12:16:10] INF: High-accuracy neural translation model matching (EN -> JA, KO)',
      '[12:17:40] INF: Voice synthesis completed successfully',
      '[12:18:22] INF: Audio-video muxing complete (H.264 / AAC profile)',
      '[12:18:24] INF: Pipeline succeeded. Job output generated.'
    ],
    steps: [
      { name: 'Caption OCR', description: 'Extracted text segments using layout scan.', status: 'completed', duration: '10s' },
      { name: 'Narration Classification', description: 'Identified main speaker profile.', status: 'completed', duration: '6s' },
      { name: 'Global Summary', description: 'Contextual summary generated.', status: 'completed', duration: '9s' },
      { name: 'Segment Translation', description: 'Translated EN -> JA, KO. 100% confidence.', status: 'completed', duration: '35s' },
      { name: 'SRT Generation', description: 'Subtitles generated successfully.', status: 'completed', duration: '4s' },
      { name: 'Voice Cloning', description: 'Cloned narrator voice successfully.', status: 'completed', duration: '1m 12s' },
      { name: 'Audio Alignment', description: 'Aligned audio timelines.', status: 'completed', duration: '8s' },
      { name: 'Video Rendering', description: 'Rendered video with dubbed audio.', status: 'completed', duration: '22s' }
    ]
  },
  {
    id: 'JOB-8290',
    sourceMedia: 'Marketing_Broll_Raw.mp4',
    sourceLang: 'EN',
    targetLangs: ['PT'],
    status: 'FAILED',
    overallProgress: 15,
    currentStep: 'ERR: Audio isolation timeout',
    createdText: '5h ago',
    createdAt: Date.now() - 300 * 60 * 1000,
    sourceSize: '198 MB',
    pipelineMode: 'OCR',
    ttsModel: 'VieNeu-TTS Turbo (Recommended)',
    keepOriginalAudio: false,
    targetVoiceSampleName: 'marketing_spokesperson.wav',
    targetVoiceSampleType: 'upload',
    logs: [
      '[09:02:10] INF: Initializing pipeline for JOB-8290',
      '[09:02:11] INF: Extracting audio tracks... OK (1 channel found)',
      '[09:02:30] INF: Caption OCR completed',
      '[09:03:45] ERR: Audio isolation model timeout while segmenting background noise',
      '[09:03:46] ERR: Pipeline aborted. Please verify input audio volume levels and retry.'
    ],
    steps: [
      { name: 'Caption OCR', description: 'Extracted text from frame overlays.', status: 'completed', duration: '19s' },
      { name: 'Narration Classification', description: 'Failed to segment narration background tracks.', status: 'active', duration: 'Failed' },
      { name: 'Global Summary', description: 'Pending', status: 'pending' },
      { name: 'Segment Translation', description: 'Pending', status: 'pending' },
      { name: 'SRT Generation', description: 'Pending', status: 'pending' },
      { name: 'Voice Cloning', description: 'Pending', status: 'pending' },
      { name: 'Audio Alignment', description: 'Pending', status: 'pending' },
      { name: 'Video Rendering', description: 'Pending', status: 'pending' }
    ]
  },
  {
    id: 'JOB-8295',
    sourceMedia: 'Tutorial_Series_Ep1.mp4',
    sourceLang: 'EN',
    targetLangs: ['IT', 'NL'],
    status: 'QUEUED',
    overallProgress: 0,
    currentStep: 'Waiting for resources',
    createdText: 'Just now',
    createdAt: Date.now() - 30 * 1000,
    sourceSize: '45 MB',
    pipelineMode: 'VLM',
    ttsModel: 'VieNeu-TTS Turbo (Recommended)',
    keepOriginalAudio: true,
    targetVoiceSampleName: 'instructor_mic_record',
    targetVoiceSampleType: 'mic',
    logs: [
      '[07:55:00] INF: Job queued. Waiting for an available GPU rendering slot...',
    ],
    steps: [
      { name: 'Caption OCR', description: 'Pending resource allocation.', status: 'pending' },
      { name: 'Narration Classification', description: 'Pending resource allocation.', status: 'pending' },
      { name: 'Global Summary', description: 'Pending resource allocation.', status: 'pending' },
      { name: 'Segment Translation', description: 'Pending resource allocation.', status: 'pending' },
      { name: 'SRT Generation', description: 'Pending resource allocation.', status: 'pending' },
      { name: 'Voice Cloning', description: 'Pending resource allocation.', status: 'pending' },
      { name: 'Audio Alignment', description: 'Pending resource allocation.', status: 'pending' },
      { name: 'Video Rendering', description: 'Pending resource allocation.', status: 'pending' }
    ]
  }
];

export const simulationLogLines = [
  'INF: Analyzing visual layout of frames for OCR correlation',
  'INF: Matching speech chunks with video timeline stamps',
  'INF: Translation progress block index #4 is green',
  'INF: Re-generating voice wave frequencies matching speaker formants',
  'INF: Running high-speed neural TTS synthesizer',
  'INF: Aligning syllables with millisecond offsets',
  'INF: Applying noise gate filters and ambient ducking parameters',
  'INF: Running audio-video track multiplexer',
  'INF: Finalizing output containers and manifest'
];
