import React from 'react';
import { Flag, Mic, MicOff, Settings, LogOut } from 'lucide-react'; // Added LogOut icon
import { Badge } from "@/components/ui/badge";

interface AgentHeaderProps {
  agentName: string;
  ticketId: string;
  callDuration: string;
  riskLevel: 'Low Risk' | 'Medium Risk' | 'High Risk';
  isMuted: boolean;
  onToggleMute: () => void;
  onEndCall: () => void;
  onSettings: () => void;
  onFlag: () => void;
  onLogout: () => void; // Added logout prop
}

const getRiskColor = (risk: string) => {
  switch (risk) {
    case 'Low Risk':
      return 'bg-green-200 text-black';
    case 'Medium Risk':
      return 'bg-risk-medium text-black';
    case 'High Risk':
      return 'bg-risk-high text-white';
    default:
      return 'bg-green-200 text-black';
  }
};

const AgentHeader: React.FC<AgentHeaderProps> = ({
  agentName,
  ticketId,
  callDuration,
  riskLevel,
  isMuted,
  onToggleMute,
  onSettings,
  onFlag,
  onLogout // Added logout prop
}) => {
  return (
    <div className="w-full flex justify-between items-center px-4 py-3 bg-white shadow">
      <div className="flex items-center space-x-4">
        <h2 className="font-semibold text-lg">{agentName}</h2>
        <Badge variant="outline" className="px-3 py-1 border border-gray-300">
          {ticketId}
        </Badge>
        <div className="flex items-center space-x-1">
          <span className="text-sm font-medium">{callDuration}</span>
        </div>
        <Badge className={`px-3 py-1 ${getRiskColor(riskLevel)}`}>
          <span className="mr-1.5 inline-block w-2 h-2 rounded-full bg-current"></span>
          {riskLevel}
        </Badge>
      </div>

      <div className="flex items-center space-x-4">
        <button
          className={`flex items-center gap-2 px-3 py-1.5 rounded-md ${isMuted ? 'bg-red-100 text-red-600 hover:bg-red-200' : 'bg-green-100 text-green-600 hover:bg-green-200'}`}
          onClick={onToggleMute}
        >
          {isMuted ?
            <>
              <MicOff className="w-4 h-4" />
              <span className="text-sm font-medium">Muted</span>
            </> :
            <>
              <Mic className="w-4 h-4" />
              <span className="text-sm font-medium">Live</span>
            </>
          }
        </button>
        <button
          className="p-2 hover:bg-gray-100 rounded-full"
          onClick={onSettings}
        >
          <Settings className="w-5 h-5" />
        </button>
        <button
          className="p-2 hover:bg-gray-100 rounded-full"
          onClick={onFlag}
        >
          <Flag className="w-5 h-5" />
        </button>
        <button
          className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-red-100 text-red-600 hover:bg-red-200 transition-colors font-semibold"
          onClick={onLogout}
          title="Logout"
        >
          <LogOut className="w-5 h-5" />
          <span className="text-sm font-medium">Logout</span>
        </button>
      </div>
    </div>
  );
};

export default AgentHeader;
