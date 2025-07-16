import React from 'react';

interface AgentProfileProps {
  name: string;
  email: string;
  employeeId: string;
}

const AgentProfile: React.FC<AgentProfileProps> = ({ name, email, employeeId }) => {
  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <div className="mb-2">
        <h3 className="font-medium text-lg">{name}</h3>
      </div>
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-sm">{email}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm">Employee ID: {employeeId}</span>
        </div>
      </div>
    </div>
  );
};

export default AgentProfile;
