import React from 'react';
import { EmotionSegment } from './emotionUtils';

interface EmotionTimelineProps {
  segments: EmotionSegment[];
  currentTime: number;
  timelineOffset: number;
  visibleDuration: number;
  isLive: boolean;
  isEmployeeSentiment?: boolean;
  getEmotionColor: (sentiment: string) => string;
}

const EmotionTimeline: React.FC<EmotionTimelineProps> = ({
  segments,
  currentTime,
  timelineOffset,
  visibleDuration,
  isLive,
  isEmployeeSentiment = false,
  getEmotionColor
}) => {
  const totalDuration = segments.reduce((acc, segment) => acc + segment.duration, 0);

  return (
    <div className="relative overflow-hidden border border-gray-200 rounded-md">
      <div 
        className="flex h-16 transition-transform duration-1000 ease-linear w-full"
        style={{ 
          transform: `translateX(-${(timelineOffset / visibleDuration) * 100}%)`,
          width: '100%'
        }}
      >
        {segments.map((segment, index) => {
          const segmentWidth = (segment.duration / visibleDuration) * 100; // Use visibleDuration for width
          const isActive = isLive && currentTime >= segment.startTime && currentTime < segment.startTime + segment.duration;
          const isPast = isLive && currentTime > segment.startTime + segment.duration;
          
          // Use white color for employee sentiment analysis
          const backgroundColor = isEmployeeSentiment ? '#ffffff' : getEmotionColor(segment.emotion);
          
          return (
            <div 
              key={index}
              className={`relative flex items-center justify-center text-xs font-medium ${
                isEmployeeSentiment ? 'text-gray-600' : 'text-white'
              } transition-all duration-500 ${
                isActive ? 'ring-2 ring-blue-400 scale-105 z-10 shadow-lg' : ''
              }`}
              style={{ 
                width: `${segmentWidth}%`,
                backgroundColor,
                opacity: isPast ? 0.7 : 1,
                minWidth: '30px',
                border: isEmployeeSentiment ? '1px solid #e5e7eb' : 'none'
              }}
              title={segment.text}
            >
              <span className="capitalize text-center px-1 truncate">
                {isEmployeeSentiment ? '' : segment.emotion}
              </span>
              {isActive && !isEmployeeSentiment && (
                <div className="absolute inset-0 bg-white bg-opacity-20 animate-pulse"></div>
              )}
              
              <div className={`absolute right-0 top-0 h-full w-px ${
                isEmployeeSentiment ? 'bg-gray-300' : 'bg-white bg-opacity-30'
              }`}></div>
            </div>
          );
        })}
      </div>
      
      {/* Live progress indicator - only for agent sentiment */}
      {isLive && segments.length > 0 && !isEmployeeSentiment && (
        <div 
          className="absolute top-0 w-1 h-16 bg-green-600 opacity-90 transition-all duration-1000 ease-linear z-20"
          style={{ 
            left: `${Math.min(((currentTime - timelineOffset) / visibleDuration) * 100, 100)}%`,
            transform: 'translateX(-50%)'
          }}
        >
          <div className="absolute -top-1 -left-1 w-3 h-3 bg-green-600 rounded-full border-2 border-white"></div>
        </div>
      )}
    </div>
  );
};

export default EmotionTimeline;