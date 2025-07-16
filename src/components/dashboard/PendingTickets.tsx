import React from 'react';
import { Badge } from "@/components/ui/badge";

interface Ticket {
  id: string;
  description: string;
  status: 'urgent' | 'inprogress' | 'pending' | string;
  urgency?: string;
}

interface PendingTicketsProps {
  tickets: Ticket[];
  loading?: boolean;
  error?: string | null;
}

const getUrgencyBadge = (urgency?: string) => {
  if (!urgency || urgency === 'Unknown') return null;
  switch (urgency) {
    case 'Critical':
      // Light red: #fdecea, text: #b71c1c, bold
      return <Badge style={{ backgroundColor: '#fdecea', color: '#b71c1c', fontWeight: 'bold' }}>Critical</Badge>;
    case 'High':
      // Light orange: #fff4e5, text: #b26a00, bold
      return <Badge style={{ backgroundColor: '#fff4e5', color: '#b26a00', fontWeight: 'bold' }}>High</Badge>;
    case 'Moderate':
      // Light yellow: #fffbe6, text: #8a6d1b, bold
      return <Badge style={{ backgroundColor: '#fffbe6', color: '#8a6d1b', fontWeight: 'bold' }}>Moderate</Badge>;
    default:
      return <Badge style={{ fontWeight: 'bold' }}>{urgency}</Badge>;
  }
};

const getStatusBadge = (status: string, urgency?: string) => {
  // Prefer urgency badge if present
  const urgencyBadge = getUrgencyBadge(urgency);
  if (urgencyBadge) return urgencyBadge;
  switch (status) {
    case 'urgent':
      return <Badge className="bg-label-red">Urgent</Badge>;
    case 'inprogress':
      return <Badge className="bg-label-yellow text-black">In Progress</Badge>;
    case 'pending':
      return <Badge className="bg-slate-200 text-slate-800">Pending</Badge>;
    default:
      return <Badge>Unknown</Badge>;
  }
};

const PendingTickets: React.FC<PendingTicketsProps & { onSelectTicket?: (ticket: Ticket) => void, selectedTicketId?: string }> = ({ tickets, loading, error, onSelectTicket, selectedTicketId }) => {
  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <h3 className="font-medium mb-3">Pending Tickets</h3>
      {loading ? (
        <div className="text-sm text-gray-500">Loading tickets...</div>
      ) : error ? (
        <div className="text-sm text-red-500">{error}</div>
      ) : tickets.length === 0 ? (
        <div className="text-sm text-gray-500">No pending tickets.</div>
      ) : (
        <div className="space-y-3">
          {tickets.map(ticket => (
            <button
              key={ticket.id}
              className={`w-full text-left flex justify-between items-center border-b pb-3 last:border-0 last:pb-0 rounded transition-colors ${selectedTicketId === ticket.id ? 'bg-blue-50' : 'hover:bg-gray-50'}`}
              style={{ cursor: 'pointer' }}
              onClick={() => onSelectTicket && onSelectTicket(ticket)}
            >
              <div>
                <p className="text-sm font-medium">{ticket.id}</p>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default PendingTickets;
