import React from 'react';
import { EmotionSegment } from './emotionUtils';

// Local helper for emotion color
const getEmotionColor = (sentiment: string) => {
  switch (sentiment) {
    case 'positive':
      return '#22c55e'; // green
    case 'negative':
      return '#ef4444'; // red
    case 'neutral':
    default:
      return '#eab308'; // yellow
  }
};

interface CurrentEmotionDisplayProps {
  currentSegment: EmotionSegment | undefined;
  currentTime: number;
  speaker: 'agent' | 'customer';
  isLive: boolean;
}

const CurrentEmotionDisplay: React.FC<CurrentEmotionDisplayProps> = ({
  currentSegment,
  currentTime,
  speaker,
  isLive
}) => {
  if (!isLive || !currentSegment) {
    return null;
  }

  return (
    <div className="bg-gray-50 rounded-lg p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-600">Current {speaker}:</span>
        <span 
          className="text-xs font-medium capitalize px-2 py-1 rounded-full text-white"
          style={{ backgroundColor: getEmotionColor(currentSegment.emotion) }}
        >
          {currentSegment.emotion}
        </span>
      </div>
      <div className="text-sm text-gray-800 italic">
        {currentSegment.text && currentSegment.text.trim() !== '' ? currentSegment.text : <span className="text-gray-400">No transcript</span>}
      </div>
      <div className="text-xs text-gray-500">
        Time: {Math.floor(currentTime / 60).toString().padStart(2, '0')}:
        {(currentTime % 60).toString().padStart(2, '0')}
      </div>
    </div>
  );
};

export default CurrentEmotionDisplay;