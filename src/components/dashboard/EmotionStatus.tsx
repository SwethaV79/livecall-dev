
import React from 'react';

interface EmotionStatusProps {
  currentTime: number;
  segmentsLength: number;
  isLive: boolean;
}

const EmotionStatus: React.FC<EmotionStatusProps> = ({
  currentTime,
  segmentsLength,
  isLive
}) => {
  if (!isLive) {
    return null;
  }

  return (
    <div className="text-xs text-gray-400 bg-gray-50 p-2 rounded">
      <div>Next emotion in: {20 - (currentTime % 20)} seconds</div>
      <div>Total segments: {segmentsLength}</div>
    </div>
  );
};

export default EmotionStatus;
