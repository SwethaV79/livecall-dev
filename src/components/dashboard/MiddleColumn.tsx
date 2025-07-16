import React from 'react';
import VoiceAnalysisGraphs from '@/components/dashboard/VoiceAnalysisGraphs';
import TimeAnalysis from '@/components/dashboard/TimeAnalysis';
import QuickResponses from '@/components/dashboard/QuickResponses';
import TicketInformation from '@/components/dashboard/TicketInformation';

interface MiddleColumnProps {
  pitchData: Array<{ time: string; value: number }>;
  energyData: Array<{ time: string; value: number }>;
  speakingRateData: Array<{ time: string; value: number }>;
  emotion: string;
  transcriptEmotions: Array<{
    emotion: string;
    timestamp: string;
    speaker: 'agent' | 'customer';
  }>;
  employeeTranscriptEmotions?: Array<{
    emotion: string;
    timestamp: string;
    speaker: 'agent' | 'customer' | 'employee';
    text?: string;
  }>;
  suggestions: string[];
  onCopySuggestion: (suggestion: string) => void;
  isMuted?: boolean;
  pendingTickets: TicketInfo[];
  selectedTicket: TicketInfo | null;
}

interface TicketInfo {
  id: string;
  description: string;
  urgency?: string;
}

const MiddleColumn: React.FC<MiddleColumnProps> = ({
  pitchData,
  energyData,
  speakingRateData,
  emotion,
  transcriptEmotions,
  employeeTranscriptEmotions = [],
  suggestions,
  onCopySuggestion,
  isMuted = false,
  pendingTickets = [],
  selectedTicket
}) => {
  // Ensure status is a valid value for CallSummary component
  const getValidStatus = (status: string): 'Resolved' | 'In Progress' | 'Requires Escalation' => {
    switch (status) {
      case 'Resolved':
      case 'In Progress':
      case 'Requires Escalation':
        return status as 'Resolved' | 'In Progress' | 'Requires Escalation';
      default:
        return 'In Progress'; // Default fallback
    }
  };

  return (
    <div className="col-span-7 space-y-4">
      {/* Voice Analysis Graphs */}
      <VoiceAnalysisGraphs 
        pitchData={pitchData}
        energyData={energyData}
        speakingRateData={speakingRateData}
        emotion={emotion}
      />
      
      {/* Live Sentiment Analysis Gantt Charts */}
      <div className="grid grid-cols-2 gap-4">
        <TimeAnalysis 
          title="Agent Sentiment Analysis" 
          emotionData={transcriptEmotions}
          isLive={true}
          isMuted={isMuted}
        />
        <TimeAnalysis 
          title="Employee Sentiment Analysis" 
          emotionData={
            employeeTranscriptEmotions
              .filter(e => e.speaker === "employee")
              .map(e => ({
                ...e,
                speaker: "customer" as "customer" // Cast to satisfy TimeAnalysis
              }))
          }
          isLive={true}
          isMuted={isMuted}
        />
      </div>
      
      {/* Quick Responses */}
      <QuickResponses 
        suggestions={suggestions}
        onCopySuggestion={onCopySuggestion}
      />

      {/* Ticket Information Module */}
      <TicketInformation ticket={selectedTicket} />
    </div>
  );
};

export default MiddleColumn;