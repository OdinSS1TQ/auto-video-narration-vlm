import React, { useEffect, useRef } from 'react';
import { ArrowLeft, Play, ExternalLink, XCircle, Code, Check, Terminal, Download, ShieldCheck } from 'lucide-react';
import { Job } from '../types';
import { api } from '../api';

interface JobDetailViewProps {
  job: Job;
  onGoBack: () => void;
  onCancelJob: (jobId: string) => void;
}

export default function JobDetailView({ job, onGoBack, onCancelJob }: JobDetailViewProps) {
  const terminalContainerRef = useRef<HTMLDivElement>(null);

  // Auto scroll terminal logs within container only (prevents scroll-jacking the window)
  useEffect(() => {
    if (terminalContainerRef.current) {
      terminalContainerRef.current.scrollTop = terminalContainerRef.current.scrollHeight;
    }
  }, [job.logs]);

  // Determine current active step index
  const activeStepIndex = job.steps.findIndex((step) => step.status === 'active');

  return (
    <div className="flex-1 flex flex-col min-h-screen bg-[#0d0d0f] text-[#e5e1e4] p-8 pb-16">
      {/* Back Header Nav */}
      <div className="flex items-center gap-4 mb-6">
        {job.status !== 'PROCESSING' && job.status !== 'QUEUED' ? (
          <button
            onClick={onGoBack}
            className="p-2 rounded-md hover:bg-[#1c1b1d] border border-[#1c1b1d] transition cursor-pointer text-[#adc6ff] flex items-center justify-center"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
        ) : (
          <div className="p-2 rounded-md border border-[#1c1b1d]/30 text-gray-600 flex items-center justify-center opacity-40 cursor-not-allowed" title="Navigation is locked during dubbing">
            <ArrowLeft className="w-4 h-4" />
          </div>
        )}
        <div>
          <h1 className="font-sans font-bold text-xl tracking-tight text-[#e5e1e4] flex items-center gap-2">
            Job Detail
          </h1>
        </div>
      </div>

      {/* Main Stats Header Block */}
      <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-6 mb-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 flex-wrap">
              <h2 className="font-mono text-xl font-bold tracking-tight text-[#e5e1e4]">{job.id}</h2>
              
              {/* Badge based on status */}
              {job.status === 'PROCESSING' && (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-blue-500/10 border border-blue-500/30 text-blue-400 font-mono text-[9px] font-semibold uppercase">
                  <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-ping"></span>
                  <span>PROCESSING</span>
                </span>
              )}
              {job.status === 'COMPLETE' && (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono text-[9px] font-semibold uppercase">
                  <Check className="w-3 h-3 text-emerald-400" />
                  <span>COMPLETE</span>
                </span>
              )}
              {job.status === 'FAILED' && (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-red-500/10 border border-red-500/30 text-red-500 font-mono text-[9px] font-semibold uppercase animate-pulse">
                  <XCircle className="w-3 h-3" />
                  <span>FAILED</span>
                </span>
              )}
              {job.status === 'QUEUED' && (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#2a2a2c] border border-[#424754]/30 text-[#c2c6d6] font-mono text-[9px] font-semibold uppercase">
                  <span>QUEUED</span>
                </span>
              )}
            </div>
            
            <p className="text-xs text-[#c2c6d6] mt-2 font-mono">
              Source: <span className="text-[#adc6ff]">{job.sourceMedia}</span> | Target: <span className="text-[#adc6ff]">{job.targetLangs.join(', ')}</span>
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            {job.status === 'COMPLETE' ? (
              <>
                <a
                  href={api.getVideoDownloadUrl(job.id)}
                  download={`${job.sourceMedia.split('.')[0]}_dubbed.mp4`}
                  className="px-4 py-2 bg-emerald-500 hover:bg-emerald-500/90 text-xs font-semibold rounded-md transition text-white flex items-center gap-1.5 cursor-pointer font-mono"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download Video</span>
                </a>
                <a
                  href={api.getSrtDownloadUrl(job.id)}
                  download={`${job.sourceMedia.split('.')[0]}_vi.srt`}
                  className="px-4 py-2 bg-[#1c1b1d] hover:bg-[#2a2a2c] text-xs font-semibold rounded-md border border-[#424754]/40 transition text-[#e5e1e4] flex items-center gap-1.5 cursor-pointer font-mono"
                >
                  <Code className="w-3.5 h-3.5" />
                  <span>Download SRT</span>
                </a>
              </>
            ) : (
              <button
                disabled={job.status === 'FAILED'}
                onClick={() => onCancelJob(job.id)}
                className={`px-4 py-2 text-xs font-semibold rounded-md transition flex items-center gap-1.5 cursor-pointer border ${
                  job.status === 'FAILED'
                    ? 'border-gray-800 text-gray-600 cursor-not-allowed'
                    : 'bg-red-500/15 hover:bg-red-500/25 text-red-400 border-red-500/20'
                }`}
              >
                <XCircle className="w-3.5 h-3.5" />
                <span>Cancel Job</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main layout contents */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        
        {/* Left pane: Overall progress & Timeline Steps (8 columns) */}
        <div className="xl:col-span-8 flex flex-col gap-6">
          
          {/* Module A: Overall Progress percent bar */}
          <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-bold font-mono text-[#c2c6d6] uppercase tracking-wider">Overall Progress</span>
              <span className="font-sans font-bold text-4xl text-[#adc6ff]">{job.overallProgress}%</span>
            </div>

            {/* Horizontal custom loader bar */}
            <div className="w-full bg-[#1c1b1d] h-3 rounded-full border border-[#1c1b1d] overflow-hidden mb-3">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  job.status === 'FAILED'
                    ? 'bg-red-500'
                    : job.status === 'COMPLETE'
                    ? 'bg-emerald-500'
                    : 'bg-gradient-to-r from-blue-600 via-[#3626ce] to-[#adc6ff] animate-pulse'
                }`}
                style={{ width: `${job.overallProgress}%` }}
              ></div>
            </div>

            <div className="flex items-center justify-between text-[11px] font-mono text-[#c2c6d6]">
              <span>Started: 14:32:01 UTC</span>
              <span>
                {job.status === 'COMPLETE' ? (
                  <span className="text-emerald-400">Pipeline Finished (100%)</span>
                ) : job.status === 'FAILED' ? (
                  <span className="text-red-400">Failed at step: {job.currentStep}</span>
                ) : job.status === 'QUEUED' ? (
                  <span>Queued in rendering pool</span>
                ) : (
                  <span>Est. Remaining: ~4m 12s</span>
                )}
              </span>
            </div>
          </div>

          {/* Module B: Pipeline steps list */}
          <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-6">
            <h3 className="font-sans font-bold text-sm tracking-tight text-[#e5e1e4] mb-4">
              Pipeline Timeline ({job.pipelineMode} Mode)
            </h3>

            <div className="flex flex-col gap-3">
              {job.steps.map((step, idx) => {
                const isStepCompleted = step.status === 'completed';
                const isStepActive = step.status === 'active';
                const isStepPending = step.status === 'pending';

                return (
                  <div
                    key={idx}
                    className={`flex items-start gap-4 p-3 rounded-md border transition-all ${
                      isStepActive
                        ? 'bg-[#1c1b1d] border-[#adc6ff]/40'
                        : isStepCompleted
                        ? 'bg-[#141416] border-[#1c1b1d]'
                        : 'bg-[#141416] border-transparent opacity-50'
                    }`}
                  >
                    {/* Circle icon marker column */}
                    <div className="pt-0.5" onClick={(e) => e.stopPropagation()}>
                      {isStepCompleted ? (
                        <div className="w-5 h-5 rounded-full bg-emerald-500/15 border border-emerald-500 text-emerald-500 flex items-center justify-center font-bold text-[10px]">
                          ✓
                        </div>
                      ) : isStepActive ? (
                        <div className="w-5 h-5 rounded-full border border-[#adc6ff] bg-[#adc6ff]/10 text-[#adc6ff] flex items-center justify-center relative">
                          <span className="w-2 h-2 bg-blue-400 rounded-full animate-ping absolute"></span>
                          <span className="w-1.5 h-1.5 bg-[#adc6ff] rounded-full"></span>
                        </div>
                      ) : (
                        <div className="w-5 h-5 rounded-full border border-[#424754] text-gray-500 flex items-center justify-center text-[10px] font-mono">
                          {idx + 1}
                        </div>
                      )}
                    </div>

                    {/* Step details content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <span
                          className={`text-xs font-semibold ${
                            isStepActive ? 'text-[#adc6ff]' : isStepCompleted ? 'text-gray-200' : 'text-gray-400'
                          }`}
                        >
                          {step.name}
                        </span>
                        
                        {step.duration && (
                          <span className="font-mono text-[10px] text-[#adc6ff]">
                            {step.duration}
                          </span>
                        )}
                      </div>

                      <p className="text-[11px] text-[#c2c6d6] mt-0.5 leading-relaxed">
                        {step.description}
                      </p>

                      {/* Mini progress bar inside active cloning step */}
                      {isStepActive && job.status === 'PROCESSING' && (
                        <div className="mt-3 bg-[#0d0d0f] rounded-lg p-2.5 border border-[#1c1b1d]">
                          <div className="flex items-center justify-between text-[9px] font-mono text-[#adc6ff] mb-1.5">
                            <span>Analyzing vocal signatures (Neural Map)...</span>
                            <span className="animate-pulse">Active Slot</span>
                          </div>
                          <div className="w-full bg-[#1c1b1d] h-1.5 rounded-full overflow-hidden">
                            <div className="bg-[#adc6ff] h-full duration-1000 animate-pulse" style={{ width: '40%' }}></div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right pane: Terminal Engine Output Logs (4 columns) */}
        <div className="xl:col-span-4 flex flex-col gap-6 h-full">
          <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-5 flex flex-col h-[520px]">
            {/* Log Header */}
            <div className="flex items-center justify-between pb-3 border-b border-[#1c1b1d] mb-4">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-[#adc6ff]" />
                <span className="text-xs font-semibold font-mono text-[#e5e1e4] tracking-wider uppercase">
                  Engine Output Log
                </span>
              </div>
              <button
                onClick={() => {
                  const text = job.logs.join('\n');
                  navigator.clipboard.writeText(text);
                  alert('Logs copied successfully!');
                }}
                className="text-[10px] font-mono font-bold bg-[#1c1b1d] hover:bg-[#2a2a2c] border border-[#424754]/40 px-2.5 py-1 rounded text-[#adc6ff] transition cursor-pointer"
              >
                Copy Output
              </button>
            </div>

            {/* Terminal Body */}
            <div
              ref={terminalContainerRef}
              className="flex-1 bg-[#0d0d0f] rounded p-3 border border-[#1c1b1d] font-mono text-[11px] overflow-y-auto leading-relaxed text-gray-300 flex flex-col gap-1.5 select-text"
            >
              {job.logs.map((logLine, idx) => {
                let colorClass = 'text-[#c2c6d6]';
                if (logLine.includes('ERR:')) colorClass = 'text-red-400 font-bold';
                else if (logLine.includes('WRN:')) colorClass = 'text-amber-400 font-medium';
                else if (logLine.includes('OK')) colorClass = 'text-emerald-400';
                else if (logLine.includes('succeeded') || logLine.includes('succeeded')) colorClass = 'text-emerald-400 font-bold';

                return (
                  <div key={idx} className={`${colorClass} whitespace-pre-wrap breakdown-all`}>
                    {logLine}
                  </div>
                );
              })}
              
              {/* Pulsing prompt indicator if processing */}
              {job.status === 'PROCESSING' && (
                <div className="flex items-center gap-1.5 text-[#adc6ff] animate-pulse">
                  <span>&gt;</span>
                  <span className="w-1.5 h-3.5 bg-[#adc6ff]"></span>
                </div>
              )}
              
              <div />
            </div>

            {/* Terminal Footer details */}
            <div className="mt-3 flex items-center justify-between text-[9px] font-mono text-[#c2c6d6]">
              <span>PID: 8492</span>
              <span>GPU Cluster: US-WEST-4-C</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
