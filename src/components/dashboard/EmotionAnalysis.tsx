
import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  ResponsiveContainer,
  ReferenceDot
} from 'recharts';

interface EmotionAnalysisProps {
  title: string;
  data: {
    time: string;
    value: number;
    emotion: 'positive' | 'negative' | 'neutral';
  }[];
}

const getEmotionColor = (emotion: string) => {
  switch (emotion) {
    case 'positive':
      return '#4ade80'; // Green
    case 'negative':
      return '#f43f5e'; // Red
    case 'neutral':
      return '#fcd34d'; // Yellow
    default:
      return '#888888'; // Grey for silence
  }
};

const EmotionAnalysis: React.FC<EmotionAnalysisProps> = ({ title, data }) => {
  const latestPoint = data[data.length - 1];
  
  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <h3 className="font-medium mb-3">{title}</h3>
      
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center space-x-6">
          <span className="flex items-center text-sm">
            <span className="w-3 h-3 bg-green-500 rounded-full mr-1"></span>
            Positive
          </span>
          <span className="flex items-center text-sm">
            <span className="w-3 h-3 bg-red-500 rounded-full mr-1"></span>
            Negative
          </span>
          <span className="flex items-center text-sm">
            <span className="w-3 h-3 bg-yellow-400 rounded-full mr-1"></span>
            Neutral
          </span>
          <span className="flex items-center text-sm">
            <span className="w-3 h-3 bg-gray-400 rounded-full mr-1"></span>
            Silence
          </span>
        </div>
        <div className="flex items-center">
          <span className="h-2 w-2 rounded-full bg-green-500 animate-live-pulse mr-1"></span>
          <span className="text-xs text-gray-500">Live</span>
        </div>
      </div>
      
      <div className="h-[150px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="time" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Line 
              type="basis" 
              dataKey="value" 
              stroke="#33C3F0" 
              strokeWidth={2} 
              dot={false}
              isAnimationActive={true}
              animationDuration={500}
            />
            <ReferenceDot 
              x={latestPoint.time} 
              y={latestPoint.value} 
              r={4}
              fill={getEmotionColor(latestPoint.emotion)}
              stroke="#33C3F0"
              strokeWidth={2}
              className="animate-chart-move"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default EmotionAnalysis;
