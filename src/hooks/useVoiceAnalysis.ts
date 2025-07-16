import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface VoiceData {
  time: string;
  pitch: {
    value: number;
    raw: number;
  };
  energy: {
    value: number;
    raw: number;
  };
  speakingRate: {
    value: number;
    raw: number;
  };
  emotion: string;
  is_speech: boolean;
}

interface EmotionData {
  emotion: string;
  prob: number;
  timestamp: string;
  speaker: 'agent' | 'customer';
}

interface TranscriptEmotionData {
  emotion: string;
  timestamp: string;
  speaker: 'agent' | 'customer';
  text?: string; // Add transcript text
}

interface VoiceAnalysisState {
  isConnected: boolean;
  isRecording: boolean;
  voiceData: VoiceData | null;
  pitchData: Array<{ time: string; value: number }>;
  energyData: Array<{ time: string; value: number }>;
  speakingRateData: Array<{ time: string; value: number }>;
  emotion: string;
  emotionData: EmotionData[];
  transcriptEmotions: TranscriptEmotionData[];
  startRecording: () => void;
  stopRecording: () => void;
  clearData: () => void;
  addTranscriptEmotion: (emotion: string, speaker: 'agent' | 'customer') => void;
}

const BACKEND_URL = 'http://localhost:5000';

export function useVoiceAnalysis(): VoiceAnalysisState & { quickResponses: string[] } {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [voiceData, setVoiceData] = useState<VoiceData | null>(null);
  const [pitchData, setPitchData] = useState<Array<{ time: string; value: number }>>([]);
  const [energyData, setEnergyData] = useState<Array<{ time: string; value: number }>>([]);
  const [speakingRateData, setSpeakingRateData] = useState<Array<{ time: string; value: number }>>([]);
  const [emotion, setEmotion] = useState('Neutral');
  const [emotionData, setEmotionData] = useState<EmotionData[]>([]);
  const [transcriptEmotions, setTranscriptEmotions] = useState<TranscriptEmotionData[]>([]);
  const [quickResponses, setQuickResponses] = useState<string[]>([]);

  // Limit data points to prevent performance issues
  const MAX_DATA_POINTS = 20;
  const MAX_EMOTION_DATA = 50;

  useEffect(() => {
    // Initialize socket connection
    const newSocket = io(BACKEND_URL);
    
    newSocket.on('connect', () => {
      console.log('Connected to voice analysis server');
      setIsConnected(true);
    });
    
    newSocket.on('disconnect', () => {
      console.log('Disconnected from voice analysis server');
      setIsConnected(false);
      setIsRecording(false);
    });
    
    // Handle graph data (separate from emotion)
    newSocket.on('graph_data', (data: VoiceData) => {
      console.log('Received graph data:', data);
      setVoiceData(data);
      
      // Update chart data
      setPitchData(prev => {
        const newData = [...prev, { time: data.time, value: data.pitch.value }];
        return newData.slice(-MAX_DATA_POINTS);
      });
      
      setEnergyData(prev => {
        const newData = [...prev, { time: data.time, value: data.energy.value }];
        return newData.slice(-MAX_DATA_POINTS);
      });
      
      setSpeakingRateData(prev => {
        const newData = [...prev, { time: data.time, value: data.speakingRate.value }];
        return newData.slice(-MAX_DATA_POINTS);
      });
    });

    // Handle emotion data from live transcription
    newSocket.on('emotion_data', (data: { emotion: string; prob: number; transcript?: string }) => {
      console.log('Received emotion data:', data);
      setEmotion(data.emotion);
      const timestamp = new Date().toISOString();
      const newEmotionData: EmotionData = {
        emotion: data.emotion,
        prob: data.prob,
        timestamp,
        speaker: 'agent' // Default to agent, can be enhanced
      };
      setEmotionData(prev => {
        const newData = [...prev, newEmotionData];
        return newData.slice(-MAX_EMOTION_DATA);
      });
      // Only add to transcriptEmotions if transcript is non-empty and different from last
      setTranscriptEmotions(prev => {
        if (!data.transcript || data.transcript.trim() === '') return prev;
        if (prev.length > 0 && prev[prev.length - 1].text === data.transcript) return prev;
        return [
          ...prev,
          {
            emotion: data.emotion,
            timestamp,
            speaker: 'agent',
            text: data.transcript || ''
          }
        ];
      });
    });

    // Handle quick response suggestions
    newSocket.on('quick_response', (data: { suggestion: string }) => {
      if (data && data.suggestion) {
        setQuickResponses(prev => [data.suggestion, ...prev].slice(0, 3)); // Keep only latest 3
      }
    });
    
    setSocket(newSocket);
    
    return () => {
      newSocket.disconnect();
    };
  }, []);

  const startRecording = () => {
    if (socket && isConnected) {
      console.log('Starting voice recording...');
      socket.emit('start_recording');
      setIsRecording(true);
    } else {
      console.log('Cannot start recording: socket not connected');
    }
  };

  const stopRecording = () => {
    if (socket && isConnected) {
      console.log('Stopping voice recording...');
      socket.emit('stop_recording');
      setIsRecording(false);
    } else {
      console.log('Cannot stop recording: socket not connected');
    }
  };

  const clearData = () => {
    setPitchData([]);
    setEnergyData([]);
    setSpeakingRateData([]);
    setEmotion('Neutral');
    setEmotionData([]);
    setTranscriptEmotions([]);
  };

  const addTranscriptEmotion = (emotion: string, speaker: 'agent' | 'customer') => {
    const timestamp = new Date().toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    });
    
    const newEmotion: TranscriptEmotionData = {
      emotion,
      timestamp,
      speaker
    };
    
    setTranscriptEmotions(prev => [...prev, newEmotion]);
  };

  return {
    isConnected,
    isRecording,
    voiceData,
    pitchData,
    energyData,
    speakingRateData,
    emotion,
    emotionData,
    transcriptEmotions,
    startRecording,
    stopRecording,
    clearData,
    addTranscriptEmotion,
    quickResponses
  };
}