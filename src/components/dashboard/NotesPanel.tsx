
import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface NotesPanelProps {
  initialNotes?: string;
  onSaveNotes: (notes: string) => void;
}

const NotesPanel: React.FC<NotesPanelProps> = ({ initialNotes = '', onSaveNotes }) => {
  const [notes, setNotes] = useState(initialNotes);
  const [expanded, setExpanded] = useState(false);
  
  const handleSaveNotes = () => {
    onSaveNotes(notes);
    setExpanded(false);
  };
  
  return (
    <div className="bg-white rounded-md border border-gray-200 p-4">
      <h3 className="font-medium mb-3">Add Notes</h3>
      
      {expanded ? (
        <>
          <Textarea 
            placeholder="Type your notes here..."
            className="min-h-[100px] mb-3"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
          
          <div className="flex space-x-2">
            <Button onClick={handleSaveNotes} className="w-full">
              Save Notes
            </Button>
            <Button 
              variant="outline" 
              onClick={() => setExpanded(false)} 
              className="w-full"
            >
              Cancel
            </Button>
          </div>
        </>
      ) : (
        <Button onClick={() => setExpanded(true)} className="w-full">
          {notes ? "Edit Notes" : "Add Notes"}
        </Button>
      )}
    </div>
  );
};

export default NotesPanel;
