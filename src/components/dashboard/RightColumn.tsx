import React from 'react';
import TranscriptPanel from '@/components/dashboard/TranscriptPanel';
import RiskPanel from '@/components/dashboard/RiskPanel';
import KnowledgeBasePanel from '@/components/dashboard/KnowledgeBasePanel';
import CompleteCallButton from '@/components/dashboard/CompleteCallButton';

interface Message {
  id: number;
  sender: 'Agent' | 'Customer';
  text: string;
  timestamp: string;
}

interface Article {
  id: number;
  title: string;
  content: string;
  category: string;
}

interface RightColumnProps {
  messages: Message[];
  riskLevel: string;
  customerTone: string;
  issueComplexity: string;
  issueComplexityLoading?: boolean;
  resolutionTime: string;
  progressValue: number;
  articles: Article[];
  onCopyLink: (id: number) => void;
  onCompleteCall: () => void;
}

const RightColumn: React.FC<RightColumnProps> = ({
  messages,
  riskLevel,
  customerTone,
  issueComplexity,
  issueComplexityLoading,
  resolutionTime,
  progressValue,
  articles,
  onCopyLink,
  onCompleteCall,
}) => {
  return (
    <div className="col-span-12 md:col-span-3 h-full flex flex-col space-y-4">
      <TranscriptPanel messages={messages} />
      
      <RiskPanel 
        riskLevel={riskLevel}
        customerTone={customerTone}
        issueComplexity={issueComplexity}
        issueComplexityLoading={issueComplexityLoading}
        resolutionTime={resolutionTime}
        progressValue={progressValue}
      />
      
      <div className="flex-1 flex flex-col space-y-4">
        <KnowledgeBasePanel 
          articles={articles}
          onCopyLink={onCopyLink}
        />
        
        <CompleteCallButton onClick={onCompleteCall} />
      </div>
    </div>
  );
};

export default RightColumn;