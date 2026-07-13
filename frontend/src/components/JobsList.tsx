import React, { useState } from 'react';
import { Search, ChevronDown, Download, Trash2, Eye, SlidersHorizontal, RefreshCw, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import { Job } from '../types';
import { api } from '../api';

interface JobsListProps {
  jobs: Job[];
  onSelectJob: (jobId: string) => void;
  onDeleteJob: (jobId: string) => void;
}

export default function JobsList({ jobs, onSelectJob, onDeleteJob }: JobsListProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedStatus, setSelectedStatus] = useState<'ALL' | 'PROCESSING' | 'COMPLETE' | 'FAILED' | 'QUEUED'>('ALL');
  const [sortOrder, setSortOrder] = useState<'NEWEST' | 'OLDEST'>('NEWEST');

  // Filter jobs based on query and status selection
  const filteredJobs = jobs.filter((job) => {
    const matchesSearch =
      job.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      job.sourceMedia.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = selectedStatus === 'ALL' || job.status === selectedStatus;
    return matchesSearch && matchesStatus;
  });

  // Sort based on selection
  const sortedJobs = [...filteredJobs].sort((a, b) => {
    if (sortOrder === 'NEWEST') {
      return b.createdAt - a.createdAt;
    } else {
      return a.createdAt - b.createdAt;
    }
  });

  return (
    <div className="flex-1 flex flex-col min-h-screen bg-[#0d0d0f] text-[#e5e1e4] p-8 pb-16">
      {/* Page Title */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="font-sans font-bold text-3xl tracking-tight text-[#e5e1e4] mb-2">Jobs</h1>
          <p className="text-sm text-[#c2c6d6]">Manage and monitor active dubbing pipelines.</p>
        </div>
      </div>

      {/* Filters bar */}
      <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg p-4 mb-6 flex flex-col md:flex-row gap-4 items-center justify-between">
        
        {/* Search Input */}
        <div className="relative w-full md:w-80">
          <Search className="absolute left-3 top-2.5 w-4 h-4 text-[#c2c6d6]" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search jobs..."
            className="w-full bg-[#1c1b1d] border border-[#1c1b1d] text-xs rounded-md pl-9 pr-4 py-2.5 text-[#e5e1e4] placeholder-[#c2c6d6] focus:outline-none focus:border-[#adc6ff] font-mono"
          />
        </div>

        {/* Buttons filters */}
        <div className="flex flex-wrap gap-2 w-full md:w-auto md:justify-end">
          {/* Status selector dropdown */}
          <div className="flex items-center gap-1.5 bg-[#1c1b1d] border border-[#1c1b1d] rounded-md px-2 py-1">
            <span className="text-[10px] uppercase tracking-wider font-mono text-[#adc6ff] pl-1">Status:</span>
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value as any)}
              className="bg-transparent text-xs text-[#e5e1e4] font-semibold border-none focus:outline-none cursor-pointer pr-4"
            >
              <option value="ALL">All States</option>
              <option value="PROCESSING">Processing</option>
              <option value="COMPLETE">Complete</option>
              <option value="FAILED">Failed</option>
              <option value="QUEUED">Queued</option>
            </select>
          </div>

          {/* Sort selection dropdown */}
          <div className="flex items-center gap-1.5 bg-[#1c1b1d] border border-[#1c1b1d] rounded-md px-2 py-1">
            <span className="text-[10px] uppercase tracking-wider font-mono text-[#adc6ff] pl-1">Sort:</span>
            <select
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value as any)}
              className="bg-transparent text-xs text-[#e5e1e4] font-semibold border-none focus:outline-none cursor-pointer pr-4"
            >
              <option value="NEWEST">Newest</option>
              <option value="OLDEST">Oldest</option>
            </select>
          </div>
        </div>
      </div>

      {/* Table Module */}
      <div className="bg-[#141416] border border-[#1c1b1d] rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-[#0e0e10] border-b border-[#1c1b1d] text-[#c2c6d6] font-mono uppercase text-[10px] tracking-wider">
                <th className="py-4 px-6 font-semibold">Job ID</th>
                <th className="py-4 px-4 font-semibold">Source Media</th>
                <th className="py-4 px-4 font-semibold">Status</th>
                <th className="py-4 px-4 font-semibold">Overall Progress</th>
                <th className="py-4 px-4 font-semibold">Current Step</th>
                <th className="py-4 px-4 font-semibold">Created</th>
                <th className="py-4 px-6 font-semibold text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1c1b1d]">
              {sortedJobs.length === 0 ? (
                <tr>
                  <td colSpan={7} className="text-center py-12 text-[#c2c6d6] bg-[#141416]">
                    No dubbing jobs found matching searching status. Create a new model to start.
                  </td>
                </tr>
              ) : (
                sortedJobs.map((job) => {
                  return (
                    <tr
                      key={job.id}
                      onClick={() => onSelectJob(job.id)}
                      className="hover:bg-[#1c1b1d]/40 transition duration-150 cursor-pointer align-middle"
                    >
                      {/* Job ID */}
                      <td className="py-4.5 px-6 font-mono font-bold text-[#adc6ff]">{job.id}</td>

                      {/* Source Media */}
                      <td className="py-4.5 px-4">
                        <div className="flex flex-col max-w-xs">
                          <span className="font-semibold text-[#e5e1e4] truncate mb-0.5">{job.sourceMedia}</span>
                          <span className="font-mono text-[10px] text-[#c2c6d6]">
                            {job.sourceLang} → {job.targetLangs.join(', ')}
                          </span>
                        </div>
                      </td>

                      {/* Status badge pill */}
                      <td className="py-4.5 px-4" onClick={(e) => e.stopPropagation()}>
                        {job.status === 'PROCESSING' && (
                          <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-blue-500/10 border border-blue-500/30 text-blue-400 font-mono text-[9px] font-semibold uppercase">
                            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-ping"></span>
                            <span>PROCESSING</span>
                          </span>
                        )}
                        {job.status === 'COMPLETE' && (
                          <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono text-[9px] font-semibold uppercase">
                            <CheckCircle className="w-3 h-3" />
                            <span>✓ COMPLETE</span>
                          </span>
                        )}
                        {job.status === 'FAILED' && (
                          <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-red-500/10 border border-red-500/30 text-red-400 font-mono text-[9px] font-semibold uppercase animate-pulse">
                            <AlertTriangle className="w-3 h-3" />
                            <span>⚠️ FAILED</span>
                          </span>
                        )}
                        {job.status === 'QUEUED' && (
                          <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-[#2a2a2c] border border-[#424754]/30 text-[#c2c6d6] font-mono text-[9px] font-semibold uppercase">
                            <Clock className="w-3 h-3" />
                            <span>QUEUED</span>
                          </span>
                        )}
                      </td>

                      {/* Overall Progress progressbar */}
                      <td className="py-4.5 px-4">
                        <div className="flex items-center gap-3">
                          <div className="w-24 bg-[#1c1b1d] h-2.5 rounded-full overflow-hidden border border-[#1c1b1d]">
                            <div
                              className={`h-full transition-all duration-500 rounded-full ${
                                job.status === 'FAILED'
                                  ? 'bg-red-500'
                                  : job.status === 'COMPLETE'
                                  ? 'bg-emerald-500'
                                  : 'bg-gradient-to-r from-indigo-500 to-blue-400'
                              }`}
                              style={{ width: `${job.overallProgress}%` }}
                            ></div>
                          </div>
                          <span className="font-mono text-[11px] font-semibold text-[#e5e1e4] w-8">
                            {job.overallProgress}%
                          </span>
                        </div>
                      </td>

                      {/* Current Step */}
                      <td className="py-4.5 px-4 font-mono text-[11px]">
                        <span
                          className={`font-medium ${
                            job.status === 'FAILED'
                              ? 'text-red-400'
                              : job.status === 'COMPLETE'
                              ? 'text-emerald-400 font-semibold'
                              : 'text-gray-300'
                          }`}
                        >
                          {job.currentStep}
                        </span>
                      </td>

                      {/* Created date */}
                      <td className="py-4.5 px-4 font-mono text-[11px] text-[#c2c6d6]">{job.createdText}</td>

                      {/* Actions */}
                      <td className="py-4.5 px-6 text-right" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-end gap-2.5">
                          {/* View details */}
                          <button
                            type="button"
                            onClick={() => onSelectJob(job.id)}
                            title="View pipeline status timeline"
                            className="p-1.5 rounded bg-[#1c1b1d] hover:bg-[#2a2a2c] text-[#c2c6d6] hover:text-[#e5e1e4] border border-[#1c1b1d] transition cursor-pointer"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>

                          {/* Conditional action */}
                          {job.status === 'COMPLETE' ? (
                            <a
                              href={api.getVideoDownloadUrl(job.id)}
                              download={`${job.sourceMedia.split('.')[0]}_dubbed.mp4`}
                              title="Download audio and video output packages"
                              className="p-1.5 rounded bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-400 border border-emerald-500/20 transition cursor-pointer inline-flex items-center justify-center"
                            >
                              <Download className="w-3.5 h-3.5" />
                            </a>
                          ) : (
                            <button
                              type="button"
                              onClick={() => {
                                if (confirm(`Are you sure you want to cancel and delete job ${job.id}?`)) {
                                  onDeleteJob(job.id);
                                }
                              }}
                              title="Discard this dubbing session job"
                              className="p-1.5 rounded bg-red-500/10 hover:bg-red-500/25 text-red-400 border border-red-500/20 transition cursor-pointer"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination mock footer info */}
        <div className="bg-[#0e0e10] px-6 py-4 border-t border-[#1c1b1d] flex items-center justify-between">
          <span className="text-xs text-[#c2c6d6] font-mono">
            Showing 1 to {sortedJobs.length} of {sortedJobs.length} results
          </span>
          <div className="flex items-center gap-1.5 font-mono text-xs">
            <button className="px-2 py-1 rounded bg-[#1c1b1d] text-[#c2c6d6] font-bold text-xs" disabled>&lt;</button>
            <button className="px-3 py-1 rounded bg-[#2a2a2c] text-[#adc6ff] font-bold text-xs">1</button>
            <button className="px-3 py-1 rounded text-[#c2c6d6] text-xs hover:bg-[#1c1b1d]">2</button>
            <button className="px-3 py-1 rounded text-[#c2c6d6] text-xs hover:bg-[#1c1b1d]">3</button>
            <span className="text-gray-600 px-1">...</span>
            <button className="px-3 py-1 rounded text-[#c2c6d6] text-xs hover:bg-[#1c1b1d]">12</button>
            <button className="px-2 py-1 rounded bg-[#1c1b1d] text-[#c2c6d6] font-bold text-xs">&gt;</button>
          </div>
        </div>
      </div>
    </div>
  );
}
