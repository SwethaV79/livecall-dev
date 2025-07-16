import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceDot,
  ReferenceLine,
  Tooltip
} from 'recharts';

interface VoiceAnalysisGraphsProps {
  pitchData: Array<{ time: string; value: number }>;
  energyData: Array<{ time: string; value: number }>;
  speakingRateData: Array<{ time: string; value: number }>;
  emotion: string;
}

const VoiceAnalysisGraphs: React.FC<VoiceAnalysisGraphsProps> = ({
  pitchData,
  energyData,
  speakingRateData,
  emotion
}) => {
  const normalize = (values: number[], minTarget: number, maxTarget: number) => {
    const min = Math.min(...values);
    const max = Math.max(...values);
    return values.map(value =>
      ((value - min) / (max - min + 1e-6)) * (maxTarget - minTarget) + minTarget
    );
  };

  const smoothData = (data: Array<{ time: string; value: number }>) => {
    return data.map((point, index, arr) => {
      const window = arr.slice(Math.max(index - 2, 0), Math.min(index + 3, arr.length));
      const avg = window.reduce((sum, p) => sum + p.value, 0) / window.length;
      return { ...point, value: Math.round(avg) };
    });
  };

  // Normalize pitch
  const rawPitchValues = pitchData.map(p => p.value);
  const normPitchValues = normalize(rawPitchValues, 50, 400);
  const adjustedPitchData = [{ time: '0', value: 0 }, ...pitchData.map((p, i) => ({ time: p.time, value: normPitchValues[i] }))];

  // Normalize energy
  const rawEnergyValues = energyData.map(p => p.value);
  const normEnergyValues = normalize(rawEnergyValues, 0, 6);
  const smoothedEnergyData = smoothData(energyData.map((p, i) => ({ time: p.time, value: normEnergyValues[i] })));
  const energyDataWithZero = [{ time: '0', value: 0 }, ...smoothedEnergyData];

  // Normalize speaking rate
  const rawSpeakingRateValues = speakingRateData.map(p => p.value);
  const normSpeakingRateValues = normalize(rawSpeakingRateValues, 50, 250);
  const smoothedSpeakingRateData = smoothData(speakingRateData.map((p, i) => ({ time: p.time, value: normSpeakingRateValues[i] })));
  const speakingRateWithZero = [{ time: '0', value: 0 }, ...smoothedSpeakingRateData];

  const renderGraph = (
    title: string,
    data: Array<{ time: string; value: number }> = [],
    color: string,
    threshold: number,
    unit: string,
    yMax: number
  ) => (
    <div className="h-[150px] relative">
      <h4 className="text-sm font-medium mb-1">{title}</h4>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 9 }}
            label={{ value: 'Time (frames)', position: 'insideBottom', offset: 0, fontSize: 10 }}
            interval={Math.floor(data.length / 5) || 1}
          />
          <YAxis
            domain={[0, yMax]}
            tick={{ fontSize: 9 }}
            label={{ value: unit, angle: -90, position: 'outsideLeft', fontSize: 10 }}
          />
          <Tooltip formatter={(value) => [`${value}`, title]} labelFormatter={(label) => `Frame: ${label}`} />
          <ReferenceLine
            y={threshold}
            stroke="red"
            strokeDasharray="5 3"
            label={{ position: 'right', value: 'Intensity Threshold', fontSize: 10 }}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            connectNulls={false}
          />
          {data.length > 0 && (
            <ReferenceDot
              x={data[data.length - 1].time}
              y={data[data.length - 1].value}
              r={4}
              fill={color}
              stroke={color}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
      <div className="absolute top-7 right-2 bg-white/70 px-1.5 py-0.5 rounded text-xs font-medium">
        Score: {data.length > 0 ? data[data.length - 1].value.toFixed(1) : '0.0'}
      </div>
    </div>
  );

  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <h3 className="font-medium mb-3">
        Real Time Voice Analysis | Detected Emotion: <span className="text-green-600">{emotion}</span>
      </h3>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-8">
          {renderGraph('Pitch', adjustedPitchData, '#33C3F0', 300, 'Pitch (Hz)', 400)}
          {renderGraph('Energy', energyDataWithZero, '#FFC107', 4.5, 'Energy (norm)', 6)}
        </div>
        {renderGraph('Speaking Rate (Words Per Minute)', speakingRateWithZero, '#4CAF50', 160, 'Speaking Rate (WPM)', 250)}

        <div className="text-center p-2 bg-gray-50 rounded-md mt-2">
          <p className="font-medium">
            Context Emotion Detected: <span className="text-green-600">{emotion}</span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default VoiceAnalysisGraphs;