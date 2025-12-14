// frontend/src/components/TextSelectionProvider.tsx
import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { useTextSelection } from '../hooks/useTextSelection';

interface TextSelectionContextType {
  selectedText: string;
  selectionPosition: { x: number; y: number } | null;
  getSelectedText: () => string;
  getSelectionPosition: () => { x: number; y: number } | null;
  clearSelection: () => void;
  consumeSelectedText: () => string;
}

const TextSelectionContext = createContext<TextSelectionContextType | undefined>(undefined);

export const TextSelectionProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const {
    selectedText,
    selectionPosition,
    getSelectedText,
    getSelectionPosition,
    clearSelection,
    consumeSelectedText
  } = useTextSelection();

  const contextValue = {
    selectedText,
    selectionPosition,
    getSelectedText,
    getSelectionPosition,
    clearSelection,
    consumeSelectedText
  };

  return (
    <TextSelectionContext.Provider value={contextValue}>
      {children}
    </TextSelectionContext.Provider>
  );
};

export const useTextSelectionContext = () => {
  const context = useContext(TextSelectionContext);
  if (context === undefined) {
    throw new Error('useTextSelectionContext must be used within a TextSelectionProvider');
  }
  return context;
};

// Standalone component that can be used to capture text selection
interface TextSelectionCaptureProps {
  onTextSelected?: (text: string) => void;
  children: ReactNode;
}

export const TextSelectionCapture: React.FC<TextSelectionCaptureProps> = ({
  onTextSelected,
  children
}) => {
  const { selectedText, updateSelection } = useTextSelection();

  // Call the callback when text is selected
  useEffect(() => {
    if (selectedText) {
      onTextSelected?.(selectedText);
    }
  }, [selectedText, onTextSelected]);

  return <>{children}</>;
};

// Popup component that appears when text is selected
interface TextSelectionPopupProps {
  onSelectAction: (action: 'explain' | 'translate' | 'save', text: string) => void;
}

export const TextSelectionPopup: React.FC<TextSelectionPopupProps> = ({ onSelectAction }) => {
  const { selectedText, selectionPosition, clearSelection } = useTextSelection();

  if (!selectedText || !selectionPosition) {
    return null;
  }

  const handleExplain = () => {
    onSelectAction('explain', selectedText);
    clearSelection();
  };

  const handleTranslate = () => {
    onSelectAction('translate', selectedText);
    clearSelection();
  };

  const handleSave = () => {
    onSelectAction('save', selectedText);
    clearSelection();
  };

  return (
    <div
      className="fixed z-50 flex space-x-1 p-1 bg-gray-800 text-white text-xs rounded shadow-lg"
      style={{
        left: selectionPosition.x,
        top: selectionPosition.y - 40 // Position above the selection
      }}
    >
      <button
        onClick={handleExplain}
        className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
        title="Explain selected text"
      >
        Explain
      </button>
      <button
        onClick={handleTranslate}
        className="px-2 py-1 bg-purple-600 hover:bg-purple-700 rounded transition-colors"
        title="Translate selected text"
      >
        Translate
      </button>
      <button
        onClick={handleSave}
        className="px-2 py-1 bg-green-600 hover:bg-green-700 rounded transition-colors"
        title="Save selected text"
      >
        Save
      </button>
      <button
        onClick={clearSelection}
        className="px-2 py-1 bg-gray-600 hover:bg-gray-700 rounded transition-colors"
        title="Close"
      >
        ×
      </button>
    </div>
  );
};