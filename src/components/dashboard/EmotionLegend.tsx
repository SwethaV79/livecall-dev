
import React from 'react';

const EmotionLegend: React.FC = () => {
  return (
    <div className="flex justify-between items-center text-xs text-gray-500">
      <span className="flex items-center">
        <span className="w-3 h-2 bg-green-500 rounded-sm mr-1"></span>
        Positive
      </span>
      <span className="flex items-center">
        <span className="w-3 h-2 bg-red-500 rounded-sm mr-1"></span>
        Negative
      </span>
      <span className="flex items-center">
        <span className="w-3 h-2 bg-yellow-500 rounded-sm mr-1"></span>
        Neutral
      </span>
      <span className="flex items-center">
        <span className="w-3 h-2 bg-gray-500 rounded-sm mr-1"></span>
        Silence
      </span>
    </div>
  );
};

export default EmotionLegend;
