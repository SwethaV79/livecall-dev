
# Call Insight Agent Dashboard

A modern, responsive React-based frontend built to support agent-customer interactions with real-time voice analysis and emotion detection.

## 🧾 Overview

This project offers a user-friendly interface for customer support agents to:

- View real-time voice analysis during customer calls (pitch, energy, speaking rate)
- Track live emotion analysis based on the context and tone of the conversation
- Access knowledge base entries during live calls
- Track call progress and sentiment trends
- Manage notes and quick responses
- Navigate using a refined and minimal UI layout

## 🛠 Tech Stack

### Frontend
- **React** (with TypeScript)
- **Vite** – lightning-fast frontend tooling
- **Tailwind CSS** – utility-first CSS framework
- **shadcn/ui** – component library
- **Recharts** - for visualization components

### Backend
- **Flask** - Python web framework for the API
- **NumPy/SciPy** - Scientific computing
- **Librosa** - Audio analysis
- **PyTorch** - For the Silero VAD model
- **SoundDevice** - Audio capture

## 📦 Project Structure

### Core Components

#### Dashboard Layout
- `Index.tsx` - Main dashboard container
- `AgentHeader.tsx` - Top navigation bar with call controls
- `CompleteCallButton.tsx` - Action button for ending calls

#### Voice Analysis
- `VoiceAnalysisGraphs.tsx` - Real-time visualization of voice metrics (pitch, energy, speaking rate)
- `EmotionAnalysis.tsx` - Emotion detection visualization
- `TimeAnalysis.tsx` - Timeline visualization of sentiment

#### Agent Tools
- `TranscriptPanel.tsx` - Conversation history in Teams-style chat
- `KnowledgeBasePanel.tsx` - Knowledge repository for quick reference
- `RiskPanel.tsx` - Risk assessment visualization
- `QuickResponses.tsx` - Templated responses for common scenarios
- `NotesPanel.tsx` - Note-taking functionality
- `CallSummary.tsx` - Overview of the current call

#### Agent Information
- `AgentProfile.tsx` - Agent details
- `CallProgress.tsx` - Call workflow stages
- `PreviousCalls.tsx` - History of past interactions
- `PendingTickets.tsx` - Upcoming work items
- `SentimentSnapshot.tsx` - At-a-glance sentiment overview

### Backend Structure
- `/backend` - Python backend for voice analysis
  - `app.py` - Flask API server
  - `voice_analyzer.py` - Voice feature extraction and analysis

## 🚀 Setup and Installation

### Frontend Setup

```bash
# Install dependencies
npm install

# Start the development server
npm run dev
```

### Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python backend/app.py
```

## 📊 Features

### Voice Analysis
The dashboard captures and displays three key voice metrics in real-time:
- **Pitch**: Frequency analysis of the agent's voice
- **Energy**: Volume and intensity of speech
- **Speaking Rate**: Words per minute calculation

### Sentiment Analysis
Tracks emotional context throughout the call:
- Color-coded timeline (positive, negative, neutral, silence)
- Overall sentiment score
- Context-aware emotion detection

### Knowledge Integration
Contextually relevant knowledge base articles are presented based on conversation:
- Quick access to solutions
- Categorized information
- Copy functionality for sharing with customers

## 🔄 Data Flow

1. Audio is captured via microphone and analyzed by the Python backend
2. Voice metrics are calculated and sent to the frontend via WebSocket
3. React components visualize the data in real-time
4. Agent can view insights and take appropriate actions

## 📱 Responsive Design
The dashboard is designed to work across different screen sizes, with a responsive grid layout that adapts to the available space.
