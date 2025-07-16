
import React from 'react';
import { Phone } from 'lucide-react';
import { Button } from "@/components/ui/button";

interface CompleteCallButtonProps {
  onClick: () => void;
}

const CompleteCallButton: React.FC<CompleteCallButtonProps> = ({ onClick }) => {
  return (
    <Button 
      className="w-full bg-green-500 hover:bg-green-600 text-white py-7" 
      onClick={onClick}
    >
      <Phone className="mr-2 w-6 h-6" />
      <span className="text-xl font-medium">Complete Call</span>
    </Button>
  );
};

export default CompleteCallButton;
