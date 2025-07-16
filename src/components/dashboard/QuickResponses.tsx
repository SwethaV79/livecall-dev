
import React from 'react';
import { Copy } from 'lucide-react';
import { Badge } from "@/components/ui/badge";

interface QuickResponsesProps {
  suggestions: string[];
  onCopySuggestion: (suggestion: string) => void;
}

const QuickResponses: React.FC<QuickResponsesProps> = ({ suggestions, onCopySuggestion }) => {
  // Only take the first 3 suggestions
  const displayedSuggestions = suggestions.slice(0, 3);
  
  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <h3 className="font-medium mb-3">Quick Responses</h3>
      
      <div className="flex flex-wrap gap-3">
        {displayedSuggestions.map((suggestion, index) => (
          <Badge 
            key={index} 
            className={`flex items-center py-2.5 px-3.5 cursor-pointer ${
              index === 0 ? 'bg-blue-100 text-blue-800' : 
              index === 1 ? 'bg-green-100 text-green-800' : 'bg-purple-100 text-purple-800'
            }`}
            onClick={() => onCopySuggestion(suggestion)}
          >
            <span className="text-sm mr-2 truncate max-w-[250px]">{suggestion}</span>
            <Copy className="h-3.5 w-3.5 ml-auto flex-shrink-0" />
          </Badge>
        ))}
      </div>
    </div>
  );
};

export default QuickResponses;
