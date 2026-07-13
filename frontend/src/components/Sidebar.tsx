import React from 'react';
import { LayoutDashboard, Briefcase, Activity, HelpCircle, LogOut, Plus, Volume2 } from 'lucide-react';

interface SidebarProps {
  currentScreen: 'new-job' | 'jobs-list' | 'job-detail' | 'system-health';
  onNavigate: (screen: 'new-job' | 'jobs-list' | 'job-detail' | 'system-health') => void;
  selectedJobId: string | null;
  isLocked?: boolean;
}

export default function Sidebar({ currentScreen, onNavigate, isLocked }: SidebarProps) {
  return (
    <div className="w-[260px] h-screen bg-[#0e0e10] border-r border-[#1c1b1d] flex flex-col justify-between fixed top-0 left-0 text-[#e5e1e4] select-none z-50">
      <div className="flex flex-col">
        {/* Brand Header */}
        <div className="p-6 pb-2 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-[#3626ce] flex items-center justify-center text-white">
            <Volume2 className="w-5 h-5 text-white animate-pulse" />
          </div>
          <div>
            <div className="font-sans font-bold text-lg tracking-tight text-[#e5e1e4]">AI Dub Studio</div>
            <div className="font-mono text-[10px] text-[#adc6ff] tracking-widest uppercase">PRO ENGINE V2.4</div>
          </div>
        </div>

        {/* Divider */}
        <div className="h-[1px] bg-[#1c1b1d] mx-6 my-4" />

        {/* Locked status banner */}
        {isLocked && (
          <div className="mx-4 mb-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-md">
            <span className="text-[10px] text-blue-400 font-mono animate-pulse block font-bold text-center">DUBBING ACTIVE</span>
            <span className="text-[9px] text-[#c2c6d6] block mt-1 leading-normal text-center">Navigation locked during video generation.</span>
          </div>
        )}

        <div className={isLocked ? 'pointer-events-none opacity-40 select-none' : ''}>
          {/* Action Button */}
          <div className="px-4 mb-4">
            <button
              id="btn-new-job"
              onClick={() => onNavigate('new-job')}
              className={`w-full py-3 px-4 rounded-md transition-all duration-200 font-medium text-sm flex items-center justify-center gap-2 cursor-pointer ${
                currentScreen === 'new-job'
                  ? 'bg-[#adc6ff] text-[#002e6a] shadow-lg shadow-[#adc6ff]/10 hover:bg-[#adc6ff]/90'
                  : 'bg-[#4d8eff]/10 text-[#adc6ff] border border-[#adc6ff]/20 hover:bg-[#4d8eff]/20'
              }`}
            >
              <Plus className="w-4 h-4" />
              <span>New Dubbing Job</span>
            </button>
          </div>

          {/* Nav list */}
          <nav className="flex flex-col gap-1 px-4">
            <button
              id="nav-dashboard"
              onClick={() => onNavigate('new-job')}
              className={`w-full text-left py-2.5 px-4 rounded-md transition-all duration-150 flex items-center gap-3 text-sm font-medium cursor-pointer ${
                currentScreen === 'new-job'
                  ? 'bg-[#2a2a2c] text-[#e5e1e4] border-l-2 border-[#adc6ff]'
                  : 'text-[#c2c6d6] hover:bg-[#1c1b1d] hover:text-[#e5e1e4]'
              }`}
            >
              <LayoutDashboard className="w-4 h-4" />
              <span>Dashboard</span>
            </button>

            <button
              id="nav-jobs"
              onClick={() => onNavigate('jobs-list')}
              className={`w-full text-left py-2.5 px-4 rounded-md transition-all duration-150 flex items-center gap-3 text-sm font-medium cursor-pointer ${
                currentScreen === 'jobs-list' || currentScreen === 'job-detail'
                  ? 'bg-[#2a2a2c] text-[#e5e1e4] border-l-2 border-[#adc6ff]'
                  : 'text-[#c2c6d6] hover:bg-[#1c1b1d] hover:text-[#e5e1e4]'
              }`}
            >
              <Briefcase className="w-4 h-4" />
              <span>Jobs</span>
            </button>

            <button
              id="nav-health"
              onClick={() => onNavigate('system-health')}
              className={`w-full text-left py-2.5 px-4 rounded-md transition-all duration-150 flex items-center gap-3 text-sm font-medium cursor-pointer ${
                currentScreen === 'system-health'
                  ? 'bg-[#2a2a2c] text-[#e5e1e4] border-l-2 border-[#adc6ff]'
                  : 'text-[#c2c6d6] hover:bg-[#1c1b1d] hover:text-[#e5e1e4]'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>System Health</span>
            </button>
          </nav>
        </div>
      </div>

      {/* Footer Navigation & Profile */}
      <div className="flex flex-col">
        {/* Divider */}
        <div className="h-[1px] bg-[#1c1b1d] mx-6 my-2" />

        <div className={`px-4 flex flex-col gap-1 pb-4 ${isLocked ? 'pointer-events-none opacity-40 select-none' : ''}`}>
          <button
            id="nav-help"
            className="w-full text-left py-2 px-4 rounded-md text-[#c2c6d6] hover:bg-[#1c1b1d] hover:text-[#e5e1e4] transition-all text-xs flex items-center gap-2.5 cursor-pointer"
          >
            <HelpCircle className="w-3.5 h-3.5" />
            <span>Help & Docs</span>
          </button>

          <button
            id="nav-logout"
            className="w-full text-left py-2 px-4 rounded-md text-[#c2c6d6] hover:bg-[#1c1b1d] hover:text-[#e5e1e4] transition-all text-xs flex items-center gap-2.5 cursor-pointer"
            onClick={() => alert("Simulated Sign Out successfully.")}
          >
            <LogOut className="w-3.5 h-3.5" />
            <span>Sign Out</span>
          </button>
        </div>

        {/* Profile Card */}
        <div className="p-4 bg-[#141416] border-t border-[#1c1b1d] flex items-center gap-3">
          <div className="relative">
            <img
              src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&q=80&w=100&h=100"
              alt="Admin"
              referrerPolicy="no-referrer"
              className="w-9 h-9 rounded-full object-cover border border-[#424754]"
            />
            <div className="absolute bottom-0 right-0 w-2.5 h-2.5 bg-emerald-500 rounded-full border-2 border-[#141416]"></div>
          </div>
          <div className="flex flex-col min-w-0">
            <span className="text-xs font-semibold truncate text-[#e5e1e4]">Admin User</span>
            <span className="font-mono text-[9px] text-[#adc6ff] tracking-wider uppercase truncate">WORKSPACE OWNER</span>
          </div>
        </div>
      </div>
    </div>
  );
}
