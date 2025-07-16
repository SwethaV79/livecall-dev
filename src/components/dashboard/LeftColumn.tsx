import React from 'react';
import AgentProfile from '@/components/dashboard/AgentProfile';
import CallProgress from '@/components/dashboard/CallProgress';
import SentimentSnapshot from '@/components/dashboard/SentimentSnapshot';
import PreviousCalls from '@/components/dashboard/PreviousCalls';
import PendingTickets from '@/components/dashboard/PendingTickets';
import NotesPanel from '@/components/dashboard/NotesPanel';

interface ProgressStep {
  id: string;
  label: string;
  checked: boolean;
}

interface PreviousCall {
  id: number;
  team: string;
  duration: string;
}

interface PendingTicket {
  id: string;
  description: string;
  status: string;
  urgency?: string;
  state?: string;
}

interface LeftColumnProps {
  agentProfile: {
    name: string;
    role: string;
    location: string;
    email: string;
    employeeId: string;
  };
  progressSteps: ProgressStep[];
  sentimentScore: number;
  previousCalls: PreviousCall[];
  pendingTickets: PendingTicket[];
  notes: string;
  onToggleStep: (id: string, checked: boolean) => void;
  onSaveNotes: (newNotes: string) => void;
  ticketsLoading?: boolean;
  ticketsError?: string | null;
  onSelectTicket?: (ticket: PendingTicket) => void;
  selectedTicketId?: string;
}

const LeftColumn: React.FC<LeftColumnProps> = ({
  agentProfile,
  progressSteps,
  sentimentScore,
  previousCalls,
  pendingTickets,
  notes,
  onToggleStep,
  onSaveNotes,
  ticketsLoading,
  ticketsError,
  onSelectTicket,
  selectedTicketId
}) => {
  // Provide fallback for ticketsLoading and ticketsError if not passed (for backward compatibility)
  const loading = typeof ticketsLoading === 'boolean' ? ticketsLoading : false;
  const error = typeof ticketsError === 'string' ? ticketsError : null;
  return (
    <div className="col-span-12 md:col-span-2 h-full flex flex-col space-y-4 overflow-hidden">
      <AgentProfile 
        name={agentProfile.name}
        email={agentProfile.email}
        employeeId={agentProfile.employeeId}
      />
      <PendingTickets 
        tickets={pendingTickets} 
        loading={loading} 
        error={error} 
        onSelectTicket={onSelectTicket}
        selectedTicketId={selectedTicketId}
      />
      <PreviousCalls calls={previousCalls} />
      <SentimentSnapshot sentimentScore={sentimentScore} />
      <CallProgress 
        steps={progressSteps}
        onToggleStep={onToggleStep}
      />
      <div className="mt-auto">
        <NotesPanel 
          initialNotes={notes}
          onSaveNotes={onSaveNotes}
        />
      </div>
    </div>
  );
};

export default LeftColumn;
