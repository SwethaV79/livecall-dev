
import React from 'react';

interface SentimentSnapshotProps {
  sentimentScore: number; // 0-100, where 0 is negative and 100 is positive
}

const SentimentSnapshot: React.FC<SentimentSnapshotProps> = ({ sentimentScore }) => {
  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <h3 className="font-medium mb-3">Post Call Sentiment Snapshot</h3>
      
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span role="img" aria-label="Sad face" className="text-xl">😞</span>
          <span role="img" aria-label="Neutral face" className="text-xl">😐</span>
          <span role="img" aria-label="Happy face" className="text-xl">😊</span>
        </div>
        
        {/* Sentiment progress bar with gradient background */}
        <div className="h-2.5 w-full bg-gradient-to-r from-red-300 via-yellow-300 to-green-300 rounded-full relative">
          {/* Pointer indicator */}
          <div 
            className="absolute top-0 w-3 h-3 rounded-full bg-blue-600 border-2 border-white -mt-[2px] shadow-sm"
            style={{ left: `${sentimentScore}%`, transform: 'translateX(-50%)' }}
          ></div>
        </div>
        
        <div className="flex justify-between text-xs">
          <span>Negative</span>
          <span>Neutral</span>
          <span>Positive</span>
        </div>
      </div>
    </div>
  );
};

export default SentimentSnapshot;
