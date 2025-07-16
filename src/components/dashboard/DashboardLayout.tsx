import React from 'react';
import AgentHeader from '@/components/dashboard/AgentHeader';

interface DashboardLayoutProps {
  agentName: string;
  ticketId: string;
  callDuration: string;
  riskLevel: 'Low Risk' | 'Medium Risk' | 'High Risk';
  isMuted: boolean;
  onToggleMute: () => void;
  onEndCall: () => void;
  onSettings: () => void;
  onFlag: () => void;
  onLogout: () => void; // Add this line
  children: React.ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  agentName,
  ticketId,
  callDuration,
  riskLevel,
  isMuted,
  onToggleMute,
  onSettings,
  onFlag,
  onLogout, // Add this line
  children,
}) => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <AgentHeader 
        agentName={agentName}
        ticketId={ticketId}
        callDuration={callDuration}
        riskLevel={riskLevel}
        isMuted={isMuted}
        onToggleMute={onToggleMute}
        onEndCall={() => {}} // Empty function since end call button is removed
        onSettings={onSettings}
        onFlag={onFlag}
        onLogout={onLogout} // Pass down to AgentHeader
      />
      
      {/* Main Content */}
      <div className="grid grid-cols-12 gap-4 p-4 h-[calc(100vh-72px)] relative">
        {children}
        
        {/* Voice Analysis Status Indicator */}
        {!isMuted && (
          <div className="absolute bottom-4 right-4 bg-green-100 text-green-700 px-3 py-1.5 rounded-full text-xs font-medium flex items-center gap-2 animate-pulse">
            <span className="h-2 w-2 bg-green-500 rounded-full inline-block"></span>
            Voice Analysis Active
          </div>
        )}
      </div>
    </div>
  );
};

export default DashboardLayout;
