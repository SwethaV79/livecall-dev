import React from 'react';
import { Progress } from "@/components/ui/progress";

interface RiskPanelProps {
  riskLevel: string;
  customerTone: string;
  issueComplexity: string;
  issueComplexityLoading?: boolean;
  resolutionTime: string;
  progressValue: number;
}

const getCustomerToneBar = (tone: string) => {
  // Lowercase for comparison
  const t = tone.toLowerCase();
  if (t === 'positive') {
    return {
      width: '75%',
      color: 'bg-green-500'
    };
  }
  if (t === 'negative') {
    return {
      width: '75%',
      color: 'bg-red-500'
    };
  }
  // Default to neutral
  return {
    width: '50%',
    color: 'bg-yellow-500'
  };
};

const getResolutionTimeBar = (resolutionTime: string) => {
  switch (resolutionTime.toLowerCase()) {
    case 'high':
      return { width: '75%', color: 'bg-red-500', textColor: 'text-red-700' };
    case 'medium':
      return { width: '50%', color: 'bg-yellow-500', textColor: 'text-yellow-700' };
    case 'low':
      return { width: '25%', color: 'bg-green-500', textColor: 'text-green-700' };
    default:
      return { width: '0%', color: 'bg-gray-300', textColor: 'text-gray-500' };
  }
};

const getIssueComplexityBar = (issueComplexity: string, loading?: boolean) => {
  if (loading) {
    return { width: '100%', color: 'bg-yellow-400 animate-pulse', text: 'Loading...' };
  }
  switch ((issueComplexity || "").toLowerCase()) {
    case 'high':
      return { width: '75%', color: 'bg-yellow-400', text: 'High' };
    case 'medium':
      return { width: '50%', color: 'bg-yellow-400', text: 'Medium' };
    case 'low':
      return { width: '25%', color: 'bg-yellow-400', text: 'Low' };
    default:
      return { width: '0%', color: 'bg-yellow-200', text: '--' };
  }
};

const RiskPanel: React.FC<RiskPanelProps> = ({
  riskLevel,
  customerTone,
  issueComplexity,
  issueComplexityLoading,
  resolutionTime,
  progressValue
}) => {
  const getRiskBadgeClass = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low risk':
        return 'bg-green-100 text-green-800';
      case 'medium risk':
        return 'bg-yellow-100 text-yellow-800';
      case 'high risk':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const toneBar = getCustomerToneBar(customerTone);
  const resolutionBar = getResolutionTimeBar(resolutionTime);
  const issueBar = getIssueComplexityBar(issueComplexity, issueComplexityLoading);

  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <h3 className="font-medium mb-3">Escalation Risk</h3>

      <div className="flex justify-between items-center mb-2">
        <span className="text-sm">Risk Level:</span>
        <span className={`text-sm px-2 py-0.5 rounded-full ${getRiskBadgeClass(riskLevel)}`}>{riskLevel}</span>
      </div>

      <div className="space-y-4 mt-4">
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm">Customer Tone</span>
            <span className="text-sm">{customerTone}</span>
          </div>
          <div className="h-2 w-full bg-gradient-to-r from-green-300 via-yellow-300 to-red-300 rounded-full">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${toneBar.color}`}
              style={{ width: toneBar.width }}
            ></div>
          </div>
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm">Issue Complexity</span>
            <span className="text-sm font-semibold capitalize text-yellow-700">
              {issueBar.text}
            </span>
          </div>
          <div className="h-2 w-full bg-yellow-100 rounded-full">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${issueBar.color}`}
              style={{ width: issueBar.width }}
            ></div>
          </div>
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm">Resolution Time</span>
            <span className={`text-sm font-semibold capitalize ${resolutionBar.textColor}`}>
              {resolutionTime ? resolutionTime : "--"}
            </span>
          </div>
          <div className="h-2 w-full bg-gradient-to-r from-green-300 via-yellow-300 to-red-300 rounded-full">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${resolutionBar.color}`}
              style={{ width: resolutionBar.width }}
            ></div>
          </div>
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm">Call Progress</span>
            <span className="text-sm">{Math.round(Math.max(progressValue, 1))}%</span>
          </div>
          <Progress value={Math.max(progressValue, 1)} className="h-2 bg-gray-200" />
        </div>
      </div>
    </div>
  );
};

export default RiskPanel;

