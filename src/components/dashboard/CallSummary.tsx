
import React from 'react';
import { Badge } from "@/components/ui/badge";

interface CallSummaryProps {
  purpose: string;
  solution: string;
  status: 'Resolved' | 'In Progress' | 'Requires Escalation';
}

const getStatusBadge = (status: string) => {
  switch (status) {
    case 'Resolved':
      return <Badge className="bg-green-200 text-green-800">Resolved</Badge>;
    case 'In Progress':
      return <Badge className="bg-yellow-200 text-yellow-800">In Progress</Badge>;
    case 'Requires Escalation':
      return <Badge className="bg-red-200 text-red-800">Requires Escalation</Badge>;
    default:
      return <Badge>Unknown</Badge>;
  }
};

const CallSummary: React.FC<CallSummaryProps> = ({ purpose, solution, status }) => {
  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <div className="flex justify-between items-center mb-3">
        <h3 className="font-medium">Call Summary</h3>
        {getStatusBadge(status)}
      </div>
      
      <div className="space-y-3">
        <div>
          <p className="text-sm text-gray-500 mb-1">Purpose of Call:</p>
          <p className="text-sm">{purpose}</p>
        </div>
        <div>
          <p className="text-sm text-gray-500 mb-1">Agent's Solution:</p>
          <p className="text-sm">{solution}</p>
        </div>
      </div>
    </div>
  );
};

export default CallSummary;
