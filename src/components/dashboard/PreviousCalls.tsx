
import React from 'react';
import { Clock } from 'lucide-react';

interface PreviousCall {
  id: number;
  team: string;
  duration: string;
}

interface PreviousCallsProps {
  calls: PreviousCall[];
}

const PreviousCalls: React.FC<PreviousCallsProps> = ({ calls }) => {
  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <h3 className="font-medium mb-3">Previous Calls</h3>
      
      <div className="space-y-2">
        {calls.map(call => (
          <div 
            key={call.id} 
            className="flex justify-between items-center border-b pb-2 last:border-0 last:pb-0"
          >
            <span className="text-sm">{call.team}</span>
            <div className="flex items-center text-sm text-gray-500">
              <Clock className="w-3 h-3 mr-1" />
              <span>{call.duration}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PreviousCalls;
