
import React, { useState } from 'react';

interface ProgressStep {
  id: string;
  label: string;
  checked: boolean;
}

interface CallProgressProps {
  steps: ProgressStep[];
  onToggleStep: (id: string, checked: boolean) => void;
}

const CallProgress: React.FC<CallProgressProps> = ({ steps, onToggleStep }) => {
  const [checkedSteps, setCheckedSteps] = useState<Set<string>>(new Set());

  const handleStepClick = (stepId: string) => {
    const newCheckedSteps = new Set(checkedSteps);
    const isChecked = !checkedSteps.has(stepId);
    
    if (isChecked) {
      newCheckedSteps.add(stepId);
    } else {
      newCheckedSteps.delete(stepId);
    }
    
    setCheckedSteps(newCheckedSteps);
    onToggleStep(stepId, isChecked);
  };

  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <h3 className="font-medium mb-3">Call Progress</h3>
      
      <div className="space-y-2">
        {steps.map(step => (
          <div key={step.id} className="flex items-center space-x-2">
            <div 
              className={`w-5 h-5 border-2 rounded cursor-pointer flex items-center justify-center transition-colors ${
                checkedSteps.has(step.id) 
                  ? 'bg-green-500 border-green-500' 
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              onClick={() => handleStepClick(step.id)}
            >
              {checkedSteps.has(step.id) && (
                <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              )}
            </div>
            <label 
              className="text-sm font-normal leading-none cursor-pointer"
              onClick={() => handleStepClick(step.id)}
            >
              {step.label}
            </label>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CallProgress;
