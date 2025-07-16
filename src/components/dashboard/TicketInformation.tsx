import React from 'react';
import { Badge } from "@/components/ui/badge";

interface TicketInfoProps {
  ticket: {
    id: string;
    urgency?: string;
    description: string;
    state?: string;
  } | null;
}

const getUrgencyBadge = (urgency?: string) => {
  if (!urgency || urgency === 'Unknown') return null;
  switch (urgency) {
    case 'Critical':
      return <Badge style={{ backgroundColor: '#fdecea', color: '#b71c1c', fontWeight: 'bold' }}>Critical</Badge>;
    case 'High':
      return <Badge style={{ backgroundColor: '#fff4e5', color: '#b26a00', fontWeight: 'bold' }}>High</Badge>;
    case 'Moderate':
      return <Badge style={{ backgroundColor: '#fffbe6', color: '#8a6d1b', fontWeight: 'bold' }}>Moderate</Badge>;
    default:
      return <Badge style={{ fontWeight: 'bold' }}>{urgency}</Badge>;
  }
};

const TicketInformation: React.FC<TicketInfoProps> = ({ ticket }) => {
  if (!ticket) return (
    <div className="bg-white rounded-md border border-gray-200 p-4 mt-4">
      <h3 className="font-medium mb-3">Ticket Information</h3>
      <div className="text-sm text-gray-500">No ticket selected.</div>
    </div>
  );
  const [comment, setComment] = React.useState("");
  return (
    <div className="bg-white rounded-md border border-gray-200 p-4 mt-4" style={{minHeight: '120px'}}>
      <h3 className="font-medium mb-3">Ticket Information</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-2">
        <div>
          <span className="text-xs text-gray-500">Incident number:</span>
          <span className="ml-2 font-semibold">{ticket.id}</span>
        </div>
        <div>
          <span className="text-xs text-gray-500">Urgency:</span>
          <span className="ml-2">{getUrgencyBadge(ticket.urgency)}</span>
        </div>
        <div>
          <span className="text-xs text-gray-500">State:</span>
          <span className="ml-2 font-semibold">{ticket.state || '-'}</span>
        </div>
        <div>
          <span className="text-xs text-gray-500">Short description:</span>
          <span className="ml-2 font-semibold">{ticket.description}</span>
        </div>
      </div>
      <textarea
        className="mt-6 w-full border border-gray-300 rounded-md p-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-200"
        rows={3}
        placeholder="Additional comments"
        value={comment}
        onChange={e => setComment(e.target.value)}
      />
      <div className="flex flex-row gap-4 mt-4">
        <button
          className="transition-all duration-150 bg-blue-600 hover:bg-blue-700 text-white font-bold py-1.5 px-3 text-xs rounded-md shadow-sm hover:shadow-md focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 transform hover:-translate-y-0.5"
          type="button"
        >
          Update
        </button>
        <button
          className="transition-all duration-150 bg-red-600 hover:bg-red-700 text-white font-bold py-1.5 px-3 text-xs rounded-md shadow-sm hover:shadow-md focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-offset-2 transform hover:-translate-y-0.5"
          type="button"
        >
          Close Ticket
        </button>
      </div>
    </div>
  );
};

export default TicketInformation;
