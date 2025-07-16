export interface EmotionSegment {
  emotion: 'positive' | 'negative' | 'neutral' | 'silence';
  duration: number;
  timestamp: string;
  startTime: number;
  text?: string;
}

// Sample conversation snippets for different emotions
export const conversationSamples = {
  agent: {
    positive: [
      "Great! I can definitely help you with that.",
      "Perfect! That solution should work perfectly.",
      "Excellent choice! You'll love this feature.",
      "Wonderful! I'm glad we could resolve this.",
      "That's fantastic! Everything looks good now."
    ],
    negative: [
      "I understand your frustration with this issue.",
      "I'm sorry to hear you're experiencing problems.",
      "That's definitely concerning, let me investigate.",
      "I apologize for the inconvenience this has caused.",
      "This is clearly not working as expected."
    ],
    neutral: [
      "Let me check the system for you.",
      "I'll need to verify a few details first.",
      "Please hold while I look into this.",
      "Let me transfer you to the right department.",
      "I'm reviewing your account information now."
    ],
    silence: [
      "Um... let me think about this...",
      "Uh, please hold on a moment...",
      "...",
      "Let me... uh... check that for you.",
      "Hold on... I'm looking..."
    ]
  },
  customer: {
    positive: [
      "Thank you so much! This is exactly what I needed.",
      "Perfect! You've been incredibly helpful.",
      "Great service! I really appreciate your help.",
      "Wonderful! That worked perfectly.",
      "Excellent! I'm very satisfied with this solution."
    ],
    negative: [
      "This is really frustrating, nothing seems to work.",
      "I'm very disappointed with this service.",
      "This problem has been going on for weeks now.",
      "I'm getting really annoyed with these issues.",
      "This is unacceptable, I need this fixed now."
    ],
    neutral: [
      "I need help with my account settings.",
      "Can you explain how this feature works?",
      "I have a question about my billing.",
      "I'd like to update my information.",
      "Could you help me understand this process?"
    ],
    silence: [
      "Um... let me see...",
      "Uh, I'm not sure about this...",
      "...",
      "Hold on... I'm thinking...",
      "Let me... uh... check something..."
    ]
  }
};

export const getRandomSample = (speaker: 'agent' | 'customer', emotion: 'positive' | 'negative' | 'neutral' | 'silence') => {
  const samples = conversationSamples[speaker][emotion];
  return samples[Math.floor(Math.random() * samples.length)];
};

export const getRandomEmotion = (): 'positive' | 'negative' | 'neutral' | 'silence' => {
  const emotions: ('positive' | 'negative' | 'neutral' | 'silence')[] = ['positive', 'negative', 'neutral', 'silence'];
  // Weight the emotions to make conversations more realistic
  const weightedEmotions = [
    ...Array(3).fill('positive'),
    ...Array(2).fill('negative'), 
    ...Array(4).fill('neutral'),
    ...Array(1).fill('silence')
  ];
  return weightedEmotions[Math.floor(Math.random() * weightedEmotions.length)];
};

// Add mapping for go_emotions to positive/negative/neutral
export const goEmotionsToSentiment = (emotion: string): 'positive' | 'negative' | 'neutral' => {
  const positive = [
    "admiration", "amusement", "approval", "caring", "curiosity", "desire", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"
  ];
  const negative = [
    "anger", "annoyance", "confusion", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"
  ];
  const neutral = [
    "realization", "surprise", "neutral"
  ];
  if (positive.includes(emotion)) return 'positive';
  if (negative.includes(emotion)) return 'negative';
  if (neutral.includes(emotion)) return 'neutral';
  return 'neutral';
};
