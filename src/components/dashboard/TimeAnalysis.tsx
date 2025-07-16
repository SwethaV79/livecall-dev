import React, { useState, useEffect, useRef } from 'react';
import { EmotionSegment, getRandomSample, getRandomEmotion } from './emotionUtils';
import EmotionLegend from './EmotionLegend';
import EmotionTimeline from './EmotionTimeline';
import CurrentEmotionDisplay from './CurrentEmotionDisplay';
import EmotionStatus from './EmotionStatus';

interface TimeAnalysisProps {
  title: string;
  emotionData?: Array<{
    emotion: string;
    timestamp: string;
    speaker: 'agent' | 'customer';
    text?: string;
  }>;
  isLive?: boolean;
  isMuted?: boolean;
}

const TimeAnalysis: React.FC<TimeAnalysisProps> = ({ title, emotionData = [], isLive = true }) => {
  const [segments, setSegments] = useState<EmotionSegment[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [timelineOffset, setTimelineOffset] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Determine speaker based on title
  const isAgent = title.toLowerCase().includes('agent');
  const speaker: 'agent' | 'customer' = isAgent ? 'agent' : 'customer';

  // Build segments from emotionData (one per transcript)
  useEffect(() => {
    if (isLive && emotionData && emotionData.length > 0) {
      const defaultDuration = 20; // seconds per segment
      // Only use entries with non-empty transcript text
      const filtered = emotionData.filter(e => e.speaker === speaker && e.text && e.text.trim() !== '');
      const segmentsFromData: EmotionSegment[] = filtered.map((e, idx, arr) => {
        const startTime = idx * defaultDuration;
        let duration = defaultDuration;
        if (idx < arr.length - 1) {
          duration = defaultDuration;
        }
        return {
          emotion: (e.emotion as 'positive' | 'negative' | 'neutral' | 'silence') || 'neutral',
          duration,
          timestamp: e.timestamp,
          startTime,
          text: e.text || ''
        };
      });
      // Only keep the last 5 segments
      const lastFive = segmentsFromData.slice(-5);
      setSegments(lastFive);
      setCurrentTime(
        lastFive.length > 0
          ? lastFive[lastFive.length - 1].startTime + lastFive[lastFive.length - 1].duration
          : 0
      );
      return;
    }

    if (!isLive) {
      // Static segments for non-live mode
      const defaultSegments: EmotionSegment[] = [
        { emotion: 'positive', duration: 15, timestamp: '00:00', startTime: 0, text: getRandomSample(speaker, 'positive') },
        { emotion: 'negative', duration: 20, timestamp: '00:15', startTime: 15, text: getRandomSample(speaker, 'negative') },
        { emotion: 'neutral', duration: 25, timestamp: '00:35', startTime: 35, text: getRandomSample(speaker, 'neutral') },
        { emotion: 'silence', duration: 10, timestamp: '01:00', startTime: 60, text: getRandomSample(speaker, 'silence') },
        { emotion: 'positive', duration: 15, timestamp: '01:10', startTime: 70, text: getRandomSample(speaker, 'positive') }
      ];
      setSegments(defaultSegments);
      setCurrentTime(0);
      return;
    }

    // Start with an initial segment
    const initialEmotion = getRandomEmotion();
    const initialSegment: EmotionSegment = {
      emotion: initialEmotion,
      duration: 20,
      timestamp: '00:00',
      startTime: 0,
      text: getRandomSample(speaker, initialEmotion)
    };
    setSegments([initialSegment]);
    setCurrentTime(0);
  }, [isLive, speaker, emotionData]);

  // Only keep the last 5 segments for display
  const visibleSegments = segments.slice(-5);

  // Find current active segment in visibleSegments
  let currentSegment = visibleSegments.find(
    s => currentTime >= s.startTime && currentTime < s.startTime + s.duration
  );
  // If no active segment (e.g., after mute/unmute), show the last segment
  if (!currentSegment && visibleSegments.length > 0) {
    currentSegment = visibleSegments[visibleSegments.length - 1];
  }

  // Calculate timeline width and positioning
  const totalDuration = visibleSegments.reduce((acc, segment) => acc + segment.duration, 0);
  const visibleDuration = Math.max(100, totalDuration); // At least 100 seconds visible

  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium">{title}</h3>
        {isLive && (
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 bg-red-500 rounded-full animate-pulse"></span>
            <span className="text-xs text-gray-500">LIVE</span>
          </div>
        )}
      </div>

      <div className="space-y-3">
        {/* Enhanced Legend */}
        <EmotionLegend />

        {/* Scrollable Live Timeline */}
        <EmotionTimeline
          segments={visibleSegments}
          currentTime={currentTime}
          timelineOffset={timelineOffset}
          visibleDuration={visibleDuration}
          isLive={isLive}
          getEmotionColor={label => {
            const green = [
              "admiration", "amusement", "approval", "caring", "desire",
              "excitement", "gratitude", "joy", "love", "optimism",
              "pride", "relief"
            ];
            const red = [
              "anger", "annoyance", "disappointment", "disapproval",
              "disgust", "embarrassment", "fear", "grief",
              "nervousness", "remorse", "sadness"
            ];
            const yellow = [
              "curiosity", "confusion", "realization", "surprise", "neutral"
            ];
            if (green.includes(label)) return '#22c55e'; // green
            if (red.includes(label)) return '#ef4444'; // red
            if (yellow.includes(label)) return '#eab308'; // yellow
            return '#eab308'; // default to yellow
          }}
        />

        {/* Timeline timestamps */}
        <div className="flex justify-between text-xs text-gray-500">
          <span>
            {Math.floor(timelineOffset / 60).toString().padStart(2, '0')}:
            {(timelineOffset % 60).toString().padStart(2, '0')}
          </span>
          <span>Current Time</span>
          <span>
            {Math.floor((timelineOffset + visibleDuration) / 60).toString().padStart(2, '0')}:
            {((timelineOffset + visibleDuration) % 60).toString().padStart(2, '0')}
          </span>
        </div>

        {/* Current transcript and emotion */}
        <CurrentEmotionDisplay
          currentSegment={currentSegment}
          currentTime={currentTime}
          speaker={speaker}
          isLive={isLive}
        />

        {/* Emotion generation status */}
        <EmotionStatus
          currentTime={currentTime}
          segmentsLength={visibleSegments.length}
          isLive={isLive}
        />
      </div>
    </div>
  );
};

export default TimeAnalysis;