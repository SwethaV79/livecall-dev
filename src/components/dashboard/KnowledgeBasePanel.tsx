import React, { useState } from 'react';
import { Copy, ChevronDown, ChevronUp } from 'lucide-react';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

interface Article {
  id: number;
  title: string;
  content: string;
  category: string;
}

interface KnowledgeBasePanelProps {
  articles: Article[];
  onCopyLink: (id: number) => void;
}

const KnowledgeBasePanel: React.FC<KnowledgeBasePanelProps> = ({ articles, onCopyLink }) => {
  return (
    <div className="bg-white rounded-md border border-gray-200 flex flex-col min-h-[400px]">
      <div className="p-4 border-b border-gray-200">
        <h3 className="font-medium text-lg">Knowledge Base</h3>
      </div>
      
      <div className="flex-1 p-4">
        <Accordion type="single" collapsible className="w-full space-y-2">
          {articles.map((article) => (
            <AccordionItem key={article.id} value={`item-${article.id}`} className="border rounded-lg">
              <AccordionTrigger className="px-4 text-sm py-3 hover:no-underline hover:bg-gray-50 rounded-t-lg">
                <div className="flex justify-between w-full items-center pr-2">
                  <span className="text-left font-medium">{article.title}</span>
                  <span className="text-xs bg-gray-100 px-2 py-1 rounded-full">{article.category}</span>
                </div>
              </AccordionTrigger>
              <AccordionContent className="px-4 pb-4">
                <div className="text-sm whitespace-pre-wrap text-gray-700 mb-3">{article.content}</div>
                <button 
                  className="flex items-center text-sm text-blue-600 hover:text-blue-800 font-medium"
                  onClick={() => onCopyLink(article.id)}
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Copy Link
                </button>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </div>
  );
};

export default KnowledgeBasePanel;