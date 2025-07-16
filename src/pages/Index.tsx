import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useToast } from "@/hooks/use-toast";
import { useVoiceAnalysis } from '@/hooks/useVoiceAnalysis';
import DashboardLayout from '@/components/dashboard/DashboardLayout';
import LeftColumn from '@/components/dashboard/LeftColumn';
import MiddleColumn from '@/components/dashboard/MiddleColumn';
import PendingTickets from '@/components/dashboard/PendingTickets';
import RightColumn from '@/components/dashboard/RightColumn';
import { useMsal } from '@azure/msal-react';
import { useNavigate } from 'react-router-dom';
import { loginRequest, msalInstance } from '@/config/msal-config';
import FloatingFooter from '@/components/common/FloatingFooter';

const Index = () => {
  // Ticket selection state for Ticket Information
  const [selectedTicket, setSelectedTicket] = useState<{
    id: string;
    description: string;
    status: string;
    urgency?: string;
    state?: string;
  } | null>(null);
  const { toast } = useToast();
  const [isMuted, setIsMuted] = useState(true); // Start muted by default
  const [callDuration, setCallDuration] = useState('00:00:00');
  const [notes, setNotes] = useState('');
  const [isCallActive, setIsCallActive] = useState(true); // Track if call is active
  const timerRef = useRef<number | null>(null);
  const [userEmail, setUserEmail] = useState('');
  const [userName, setUserName] = useState('');
  const [progressValue, setProgressValue] = useState(1); // Start at 1%
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [isPort7000Connected, setIsPort7000Connected] = useState(false);
  const [riskLevel, setRiskLevel] = useState<'Low Risk' | 'Medium Risk' | 'High Risk'>('Low Risk');
  const [resolutionTime, setResolutionTime] = useState<string>("");
  const [callStartTimestamp, setCallStartTimestamp] = useState<number | null>(null);
  const [issueComplexity, setIssueComplexity] = useState<string>("");
  const [issueComplexityLoading, setIssueComplexityLoading] = useState(false);

  // Use voice analysis hook
  const {
    isConnected,
    isRecording,
    pitchData,
    energyData,
    speakingRateData,
    emotion,
    transcriptEmotions,
    startRecording,
    stopRecording,
    clearData,
    addTranscriptEmotion,
    quickResponses
  } = useVoiceAnalysis();

  // Mock data - moved before it's used
  const messages = [
    {
      id: 1,
      sender: 'Agent' as const,
      text: 'Hello, thank you for calling support. How can I assist you today?',
      timestamp: '14:03:22'
    },
    {
      id: 2,
      sender: 'Customer' as const,
      text: 'Hi, I\'m having trouble with my laptop. It\'s running very slow and sometimes freezes.',
      timestamp: '14:03:35'
    },
    {
      id: 3,
      sender: 'Agent' as const,
      text: 'I understand that can be frustrating. Let me help you troubleshoot this issue. Can you tell me when did you first notice this problem?',
      timestamp: '14:03:50'
    },
    {
      id: 4,
      sender: 'Customer' as const,
      text: 'It started about a week ago. I installed some software updates and since then it\'s been slow.',
      timestamp: '14:04:10'
    }
  ];

  // Handle call duration
  useEffect(() => {
    if (!isCallActive) return; // Don't start timer if call is not active

    let seconds = 0;
    let minutes = 0;
    let hours = 0;

    const updateDuration = () => {
      seconds++;
      if (seconds >= 60) {
        seconds = 0;
        minutes++;
      }
      if (minutes >= 60) {
        minutes = 0;
        hours++;
      }

      const formattedHours = hours.toString().padStart(2, '0');
      const formattedMinutes = minutes.toString().padStart(2, '0');
      const formattedSeconds = seconds.toString().padStart(2, '0');

      setCallDuration(`${formattedHours}:${formattedMinutes}:${formattedSeconds}`);
    };

    // Start timer
    timerRef.current = window.setInterval(updateDuration, 1000);

    return () => {
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
      }
    };
  }, [isCallActive]);

  // Start/stop recording based on mute state
  const handleToggleMute = useCallback(() => {
    if (isConnected) {
      if (isMuted) {
        // Unmuting - start recording
        startRecording();
        toast({
          title: "Microphone Unmuted",
          description: "Voice analysis started",
          duration: 2000,
        });
        setProgressValue((prev) => prev < 1 ? 1 : prev + 1); // Start/increment progress at unmute
      } else {
        // Muting - stop recording
        stopRecording();
        toast({
          title: "Microphone Muted",
          description: "Voice analysis paused",
          duration: 2000,
        });
      }
    } else {
      toast({
        title: "Connection Error",
        description: "Not connected to voice analysis server. Please make sure the backend is running.",
        variant: "destructive",
        duration: 3000,
      });
    }

    setIsMuted(!isMuted);
  }, [isMuted, isConnected, startRecording, stopRecording, toast]);

  // Handle initial connection notification
  useEffect(() => {
    if (isConnected) {
      toast({
        title: "Voice Analysis Connected",
        description: "Real-time voice analysis is now available. Click the mic button to start.",
        duration: 3000,
      });
    }
  }, [isConnected, toast]);

  // Mock function to simulate emotion detection from transcript
  const analyzeTranscriptEmotion = (text: string, speaker: 'Agent' | 'Customer'): string => {
    const lowerText = text.toLowerCase();

    // Simple keyword-based emotion detection for demo
    if (lowerText.includes('frustrat') || lowerText.includes('slow') || lowerText.includes('problem') || lowerText.includes('trouble')) {
      return 'frustrated';
    }
    if (lowerText.includes('thank') || lowerText.includes('help') || lowerText.includes('assist') || lowerText.includes('hello')) {
      return 'happy';
    }
    if (lowerText.includes('understand') || lowerText.includes('troubleshoot') || lowerText.includes('started')) {
      return 'neutral';
    }

    return 'neutral';
  };

  // Enhanced simulation for real-time emotion detection from transcript
  useEffect(() => {
    if (isRecording) {
      // More realistic emotion sequence based on conversation flow
      const sampleEmotions = [
        { emotion: 'happy', speaker: 'agent' as const, delay: 1000 },
        { emotion: 'frustrated', speaker: 'customer' as const, delay: 3000 },
        { emotion: 'neutral', speaker: 'agent' as const, delay: 5000 },
        { emotion: 'satisfied', speaker: 'customer' as const, delay: 7000 },
        { emotion: 'happy', speaker: 'agent' as const, delay: 9000 },
        { emotion: 'neutral', speaker: 'customer' as const, delay: 11000 },
        // Additional emotions for longer conversation
        { emotion: 'annoyed', speaker: 'customer' as const, delay: 13000 },
        { emotion: 'optimism', speaker: 'agent' as const, delay: 15000 },
        { emotion: 'pleased', speaker: 'customer' as const, delay: 17000 },
      ];

      const timeouts = sampleEmotions.map(({ emotion, speaker, delay }) =>
        setTimeout(() => {
          addTranscriptEmotion(emotion, speaker);
          console.log(`Added emotion: ${emotion} for ${speaker}`);
        }, delay)
      );

      return () => {
        timeouts.forEach(timeout => clearTimeout(timeout));
      };
    }
  }, [isRecording, addTranscriptEmotion]);

  // Enhanced messages with emotion analysis - now after messages is declared
  const messagesWithEmotions = messages.map(message => {
    const detectedEmotion = analyzeTranscriptEmotion(message.text, message.sender);
    return {
      ...message,
      emotion: detectedEmotion
    };
  });

  const knowledgeArticles = [
    {
      id: 1,
      title: 'Software Installation: Steps to Consider',
      content: 'When installing new software, ensure that your system meets minimum requirements. Before installation:\n\n1. Close all other applications\n2. Run a virus scan\n3. Create a system restore point\n4. Check for sufficient disk space\n\nIf the installation fails, try running as administrator or in compatibility mode.',
      category: 'Installation'
    },
    {
      id: 2,
      title: 'System Performance Troubleshooting',
      content: 'If the system is running slow:\n\n1. Check for background processes using Task Manager\n2. Scan for malware\n3. Clean temporary files using Disk Cleanup\n4. Defragment the hard drive if it\'s not an SSD\n5. Check startup programs and disable unnecessary ones',
      category: 'Performance'
    },
    {
      id: 3,
      title: 'Common Windows Update Issues',
      content: 'If Windows updates are causing performance issues:\n\n1. Check Windows Update history for recent installations\n2. Use Windows Update Troubleshooter\n3. Consider rolling back recent updates\n4. Ensure your drivers are compatible with the latest Windows version\n5. Check system resource usage after updates',
      category: 'Updates'
    }
  ];

  const progressSteps = [
    { id: 'step1', label: 'Identity Verified', checked: true },
    { id: 'step2', label: 'Issue Identified', checked: true },
    { id: 'step3', label: 'Cause Analysis', checked: false },
    { id: 'step4', label: 'Solution Suggested', checked: false },
    { id: 'step5', label: 'Issue Fixed', checked: false },
  ];

  const previousCalls = [
    { id: 1, team: 'L1 Team', duration: '8 minutes' },
    { id: 2, team: 'Technical Support', duration: '15 minutes' }
  ];


  // Pending tickets state (fetched from backend)
  const [pendingTickets, setPendingTickets] = useState<{
    id: string;
    description: string;
    status: string;
    urgency?: string;
    state?: string;
  }[]>([]);
  const [ticketsLoading, setTicketsLoading] = useState(false);
  const [ticketsError, setTicketsError] = useState<string | null>(null);

  // Fetch pending tickets from backend API
  useEffect(() => {
    const fetchTickets = async () => {
      setTicketsLoading(true);
      setTicketsError(null);
      try {
        const res = await fetch('/api/pending-tickets');
        if (!res.ok) throw new Error('Failed to fetch tickets');
        const data = await res.json();
        setPendingTickets(data.tickets || []);
        // Always update selectedTicket to the latest object from API if it exists
        if ((data.tickets || []).length > 0) {
          // If a ticket is already selected, find the updated object by id
          if (selectedTicket) {
            const updated = data.tickets.find((t: any) => t.id === selectedTicket.id);
            setSelectedTicket(updated || data.tickets[0]);
          } else {
            setSelectedTicket(data.tickets[0]);
          }
        } else {
          setSelectedTicket(null);
        }
      } catch (err: any) {
        setTicketsError(err.message || 'Error fetching tickets');
      } finally {
        setTicketsLoading(false);
      }
    };
    fetchTickets();
    // Optionally, poll every 30s for real-time updates
    const interval = setInterval(fetchTickets, 30000);
    return () => clearInterval(interval);
  }, []);

  // Generate fixed time segments for sentiment analysis (only once)
  const generateFixedTimeSegments = () => {
    const emotions = ['positive', 'negative', 'neutral', 'silence'] as const;
    const timeSegments = [];

    // Create segments with proper timestamps (00:00, 00:30, 01:00, etc.)
    for (let i = 0; i < 5; i++) {
      const minutes = Math.floor(i / 2);
      const seconds = (i % 2) * 30;
      const time = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

      const segments = [];
      let remainingPercentage = 100;

      // Create 2-4 segments for each time block
      const segmentCount = 2 + Math.floor(Math.random() * 3);
      for (let j = 0; j < segmentCount; j++) {
        const isLast = j === segmentCount - 1;
        const duration = isLast ? remainingPercentage : Math.floor(remainingPercentage / (segmentCount - j));

        const emotion = emotions[Math.floor(j % emotions.length)];
        segments.push({ emotion, duration });
        remainingPercentage -= duration;
      }

      timeSegments.push({ time, segments });
    }

    return timeSegments;
  };

  // Use useMemo to ensure these are generated only once
  const agentTimeSegment = useMemo(() => generateFixedTimeSegments(), []);
  const customerTimeSegment = useMemo(() => generateFixedTimeSegments(), []);

  // Quick response suggestions - limited to 3
  // const suggestions = [
  //   "Let me check the status of that for you.",
  //   "I understand your frustration. Let's find a solution together.",
  //   "Could you please provide more details about the issue?"
  // ];

  // Event handlers
  const handleEndCall = () => {
    stopRecording();
    toast({
      title: "Call Ended",
      description: "The call has been terminated",
      variant: "destructive",
    });
  };

  const handleSettings = () => {
    toast({
      title: "Settings",
      description: "Settings panel would open here",
    });
  };

  const handleFlag = () => {
    toast({
      title: "Call Flagged",
      description: "This call has been flagged for review",
    });
  };

  const handleToggleStep = (id: string, checked: boolean) => {
    toast({
      title: checked ? "Step Completed" : "Step Unchecked",
      description: `${id} has been ${checked ? 'marked as complete' : 'unmarked'}`,
    });
  };

  const handleCopyLink = (id: number) => {
    toast({
      title: "Link Copied",
      description: `Article #${id} link copied to clipboard`,
    });
  };

  const handleCopySuggestion = (suggestion: string) => {
    toast({
      title: "Response Copied",
      description: "Quick response copied to clipboard",
    });
  };

  const handleSaveNotes = (newNotes: string) => {
    setNotes(newNotes);
    toast({
      title: "Notes Saved",
      description: "Your notes have been saved",
    });
  };

  const handleCompleteCall = async () => {
    // Stop the timer
    setIsCallActive(false);
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
    }
    stopRecording();
    setProgressValue(100);

    // Calculate resolution time
    let durationMin = 0;
    if (callStartTimestamp) {
      durationMin = (Date.now() - callStartTimestamp) / 1000 / 60;
    }
    let resTime = "";
    if (durationMin > 7.5) {
      resTime = "High";
    } else if (durationMin > 4 && durationMin <= 7.5) {
      resTime = "Medium";
    } else if (durationMin <= 4) {
      resTime = "Low";
    } else {
      resTime = "";
    }
    setResolutionTime(resTime);

    setIssueComplexityLoading(true);
    try {
      const conversation = getConversationText();
      if (conversation.length > 0) {
        const resp = await fetch('/api/predict_severity', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ conversation })
        });
        if (resp.ok) {
          const data = await resp.json();
          let sev = (data.severity || "").toLowerCase();
          if (sev === "high") setIssueComplexity("High");
          else if (sev === "medium") setIssueComplexity("Medium");
          else if (sev === "low") setIssueComplexity("Low");
          else setIssueComplexity("");
        } else {
          setIssueComplexity("");
        }
      } else {
        setIssueComplexity("");
      }
    } catch (e) {
      setIssueComplexity("");
    } finally {
      setIssueComplexityLoading(false);
    }

    toast({
      title: "Call Completed",
      description: "The call has been marked as complete",
    });
  };

  // Utility to extract display name as per requirements
  const getDisplayName = (fullName: string) => {
    const parts = fullName.trim().split(/\s+/);
    if (parts.length === 3) {
      return parts[1]; // middle name
    }
    if (parts.length === 2) {
      return parts[0]; // first name
    }
    return fullName;
  };

  // Fetch user data on mount
  useEffect(() => {
    const fetchUserData = async () => {
      const accounts = msalInstance.getAllAccounts();
      if (accounts.length > 0) {
        const user = accounts[0];
        setUserName(user.name || ''); // Agent name (from SSO)
        setUserEmail(user.username || ''); // Agent email (from SSO)
      } else {
        setUserName('');
        setUserEmail('');
      }
    };
    fetchUserData();
  }, []);

  // Hardcoded caller info for profile module
  const callerProfile = {
    name: 'Kiran Kumar',
    email: 'kiran.kumar@wipro.com',
    employeeId: '456728',
    role: 'Customer', // Added role
    location: 'Bangalore' // Added location
  };

  // Add logout handler for MSAL
  const handleLogout = () => {
    msalInstance.logoutRedirect();
  };

  // New: State for employee transcript/emotion data
  const [employeeTranscriptEmotions, setEmployeeTranscriptEmotions] = useState<Array<{
    emotion: string;
    timestamp: string;
    speaker: 'agent' | 'customer' | 'employee';
    text?: string;
  }>>([]);

  // New: WebSocket connection for employee sentiment (ws://localhost:7000)
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:7000');
    ws.onopen = () => {
      console.log('✅ Connected to ws://localhost:7000');
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Expecting: { emotion, prob, transcript, speaker }
        if (data && data.speaker === 'employee') {
          setEmployeeTranscriptEmotions(prev => [
            ...prev,
            {
              emotion: data.emotion,
              timestamp: new Date().toISOString().slice(11, 19), // HH:MM:SS
              speaker: 'employee',
              text: data.transcript
            }
          ]);
        }
      } catch (e) {
        console.error('Error parsing employee websocket message:', e);
      }
    };
    ws.onerror = (err) => {
      console.error('Employee WebSocket error:', err);
    };
    ws.onclose = () => {
      console.log('Employee WebSocket closed');
    };
    return () => {
      ws.close();
    };
  }, []);

  // Track connection to port 7000 (employee sentiment)
  useEffect(() => {
    let ws: WebSocket | null = null;
    let didUnmount = false;
    ws = new WebSocket('ws://localhost:7000');
    ws.onopen = () => {
      if (!didUnmount) setIsPort7000Connected(true);
    };
    ws.onclose = () => {
      if (!didUnmount) setIsPort7000Connected(false);
    };
    ws.onerror = () => {
      if (!didUnmount) setIsPort7000Connected(false);
    };
    return () => {
      didUnmount = true;
      if (ws) ws.close();
    };
  }, []);

  // Progress bar logic: only increment if both ports are connected and call is unmuted/active
  useEffect(() => {
    let interval: number | undefined;
    if (
      isCallActive &&
      !isMuted &&
      isConnected && // port 5000
      isPort7000Connected // port 7000
    ) {
      interval = window.setInterval(() => {
        setElapsedSeconds((prev) => {
          if (prev >= 420) return 420;
          return prev + 1;
        });
      }, 1000);
    }
    return () => {
      if (interval) window.clearInterval(interval);
    };
  }, [isCallActive, isMuted, isConnected, isPort7000Connected]);

  // Update progressValue based on elapsedSeconds
  useEffect(() => {
    // 1% at start, 100% at 420 seconds (7 min)
    const progress = Math.min(1 + (elapsedSeconds / 420) * 99, 100);
    setProgressValue(progress);
  }, [elapsedSeconds]);

  // Reset progress if call is ended or muted
  useEffect(() => {
    if (!isCallActive || isMuted) {
      setElapsedSeconds((prev) => prev); // Pause, do not reset
    }
  }, [isCallActive, isMuted]);

  // Calculate Escalation Risk based on agent and employee/customer sentiments
  useEffect(() => {
    // Rolling window: last 5 segments (approx 1-2 minutes)
    const windowSize = 5;

    // Helper to get last N emotions for a speaker
    const getLastEmotions = (arr: any[], speaker: string) =>
      arr.filter(e => e.speaker === speaker).slice(-windowSize).map(e => (e.emotion || '').toLowerCase());

    const agentEmotions = getLastEmotions(transcriptEmotions, 'agent');
    const employeeEmotions = getLastEmotions(
      (Array.isArray(employeeTranscriptEmotions) ? employeeTranscriptEmotions : []),
      'employee'
    );

    // Count positives and negatives
    const positiveList = [
      "admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief", "happy", "satisfied", "pleased"
    ];
    const negativeList = [
      "anger", "annoyance", "confusion", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness", "frustrated", "annoyed"
    ];

    const agentNeg = agentEmotions.filter(e => negativeList.includes(e)).length;
    const agentPos = agentEmotions.filter(e => positiveList.includes(e)).length;
    const employeeNeg = employeeEmotions.filter(e => negativeList.includes(e)).length;
    const employeePos = employeeEmotions.filter(e => positiveList.includes(e)).length;

    // Trend: check if employee sentiment is getting worse (last 3)
    const employeeTrend = employeeEmotions.slice(-3);
    const trendWorse = employeeTrend.length === 3 &&
      negativeList.includes(employeeTrend[2]) &&
      (employeeTrend[0] === "positive" || employeeTrend[0] === "neutral") &&
      (employeeTrend[1] === "neutral" || negativeList.includes(employeeTrend[1]));

    // Heuristic mapping
    let newRisk: 'Low Risk' | 'Medium Risk' | 'High Risk' = 'Low Risk';

    if (
      (employeeNeg >= 2 && employeePos === 0 && agentPos === 0) ||
      (agentNeg + employeeNeg >= 3 && agentPos === 0 && employeePos === 0)
    ) {
      newRisk = 'Medium Risk';
    }
    if (
      (employeeNeg >= 3 && agentNeg >= 1) ||
      (employeeNeg >= 2 && agentEmotions.every(e => e === "neutral")) ||
      trendWorse
    ) {
      newRisk = 'High Risk';
    }
    if (
      agentNeg === 0 && employeeNeg === 0 &&
      (agentPos > 0 || employeePos > 0)
    ) {
      newRisk = 'Low Risk';
    }

    setRiskLevel(newRisk);
  }, [transcriptEmotions, employeeTranscriptEmotions]);

  // Compute customerTone from latest employee emotion (from port 7000)
  const customerTone = useMemo(() => {
    if (!employeeTranscriptEmotions || employeeTranscriptEmotions.length === 0) return "Neutral";
    // Take the last employee emotion
    const last = employeeTranscriptEmotions.filter(e => e.speaker === "employee").slice(-1)[0];
    if (!last) return "Neutral";
    // Map go_emotions to positive/negative/neutral
    const positive = [
      "admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief", "happy", "satisfied", "pleased"
    ];
    const negative = [
      "anger", "annoyance", "confusion", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness", "frustrated", "annoyed"
    ];
    const e = (last.emotion || "").toLowerCase();
    if (positive.includes(e)) return "Positive";
    if (negative.includes(e)) return "Negative";
    return "Neutral";
  }, [employeeTranscriptEmotions]);

  // Track call start time when call becomes active
  useEffect(() => {
    if (isCallActive && callStartTimestamp === null) {
      setCallStartTimestamp(Date.now());
    }
    // Reset callStartTimestamp if call is reset
    if (!isCallActive && callStartTimestamp !== null) {
      // do not reset here, only set on new call
    }
  }, [isCallActive, callStartTimestamp]);

  // Helper to combine agent and employee conversation for severity prediction
  const getConversationText = () => {
    // Combine transcriptEmotions (agent) and employeeTranscriptEmotions (employee)
    const agentLines = transcriptEmotions
      .filter(e => e.speaker === "agent" && e.text && e.text.trim() !== "")
      .map(e => `Agent: ${e.text}`);
    const employeeLines = (employeeTranscriptEmotions || [])
      .filter(e => e.speaker === "employee" && e.text && e.text.trim() !== "")
      .map(e => `Employee: ${e.text}`);
    // Interleave by timestamp order if possible, else just concatenate
    const allLines = [...agentLines, ...employeeLines].sort((a, b) => {
      // Try to sort by timestamp if available
      const getTime = (line: string) => {
        // Extract timestamp from original objects if needed
        // Not strictly necessary, as order is usually fine
        return 0;
      };
      return getTime(a) - getTime(b);
    });
    return allLines.join('\n');
  };

  return (
    <>
      <DashboardLayout
        agentName={userName}
        ticketId="INC4567"
        callDuration={callDuration}
        riskLevel={riskLevel}
        isMuted={isMuted}
        onToggleMute={handleToggleMute}
        onEndCall={handleEndCall}
        onSettings={handleSettings}
        onFlag={handleFlag}
        onLogout={handleLogout}
      >
        <LeftColumn
          agentProfile={callerProfile}
          progressSteps={progressSteps}
          sentimentScore={75}
          previousCalls={previousCalls}
          pendingTickets={pendingTickets}
          notes={notes}
          onToggleStep={handleToggleStep}
          onSaveNotes={handleSaveNotes}
          ticketsLoading={ticketsLoading}
          ticketsError={ticketsError}
          onSelectTicket={setSelectedTicket}
          selectedTicketId={selectedTicket?.id}
        />

        <MiddleColumn
          pitchData={pitchData}
          energyData={energyData}
          speakingRateData={speakingRateData}
          emotion={emotion}
          transcriptEmotions={
            transcriptEmotions.length === 0
              ? [{
                emotion: "neutral",
                timestamp: new Date().toISOString().slice(11, 19),
                speaker: "agent",
                text: "Waiting for the Agent..."
              }]
              : transcriptEmotions
          }
          suggestions={quickResponses}
          onCopySuggestion={handleCopySuggestion}
          pendingTickets={pendingTickets}
          selectedTicket={selectedTicket}
          employeeTranscriptEmotions={
            employeeTranscriptEmotions.length === 0
              ? [{
                emotion: "neutral",
                timestamp: new Date().toISOString().slice(11, 19),
                speaker: "employee",
                text: "Waiting for the Employee..."
              }]
              : employeeTranscriptEmotions
          }
        />

        <RightColumn
          messages={messages}
          riskLevel={riskLevel}
          customerTone={customerTone}
          issueComplexity={issueComplexity}
          issueComplexityLoading={issueComplexityLoading}
          resolutionTime={resolutionTime}
          progressValue={progressValue}
          articles={knowledgeArticles}
          onCopyLink={handleCopyLink}
          onCompleteCall={handleCompleteCall}
        />
      </DashboardLayout>
      <FloatingFooter />
    </>
  );
};

export default Index;

