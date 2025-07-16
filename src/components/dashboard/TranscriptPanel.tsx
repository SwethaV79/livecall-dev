
import React from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";

interface Message {
  id: number;
  sender: 'Agent' | 'Customer';
  text: string;
  timestamp: string;
}

interface TranscriptPanelProps {
  messages: Message[];
}

const TranscriptPanel: React.FC<TranscriptPanelProps> = ({ messages }) => {
  return (
    <div className="bg-white rounded-md border border-gray-200 flex flex-col h-1/3">
      <div className="p-3 border-b border-gray-200">
        <h3 className="font-medium">Conversation</h3>
      </div>
      
      <ScrollArea className="flex-1 p-3">
        <div className="space-y-4">
          {messages.map((message) => (
            <div 
              key={message.id} 
              className={`flex ${message.sender === 'Agent' ? 'justify-end' : 'justify-start'}`}
            >
              <div 
                className={`max-w-[80%] rounded-lg p-3 ${
                  message.sender === 'Agent' 
                    ? 'bg-blue-100 text-blue-900' 
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                <div className="flex justify-between items-center mb-1">
                  <span className="font-medium text-xs">{message.sender}</span>
                  <span className="text-xs text-gray-500">{message.timestamp}</span>
                </div>
                <p className="text-sm break-words">{message.text}</p>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};

export default TranscriptPanel;
