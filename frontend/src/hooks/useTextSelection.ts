// frontend/src/hooks/useTextSelection.ts
import { useState, useEffect, useCallback } from 'react';

export const useTextSelection = () => {
  const [selectedText, setSelectedText] = useState<string>('');
  const [selectionPosition, setSelectionPosition] = useState<{x: number, y: number} | null>(null);

  // Function to get the currently selected text
  const getSelectedText = useCallback((): string => {
    const selection = window.getSelection();
    return selection ? selection.toString().trim() : '';
  }, []);

  // Function to get the position of the selection
  const getSelectionPosition = useCallback((): {x: number, y: number} | null => {
    const selection = window.getSelection();
    if (!selection || selection.toString().trim() === '') {
      return null;
    }

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    return {
      x: rect.left + window.scrollX,
      y: rect.top + window.scrollY
    };
  }, []);

  // Handler for selection change
  const handleSelectionChange = useCallback(() => {
    const text = getSelectedText();
    const position = getSelectionPosition();

    setSelectedText(text);
    setSelectionPosition(position);
  }, [getSelectedText, getSelectionPosition]);

  // Effect to add and remove event listeners
  useEffect(() => {
    // Add event listeners for selection changes
    document.addEventListener('mouseup', handleSelectionChange);
    document.addEventListener('keyup', handleSelectionChange);

    // Cleanup function to remove event listeners
    return () => {
      document.removeEventListener('mouseup', handleSelectionChange);
      document.removeEventListener('keyup', handleSelectionChange);
    };
  }, [handleSelectionChange]);

  // Function to clear the current selection
  const clearSelection = useCallback(() => {
    const selection = window.getSelection();
    if (selection) {
      selection.removeAllRanges();
    }
    setSelectedText('');
    setSelectionPosition(null);
  }, []);

  // Function to get the currently selected text and reset it
  const consumeSelectedText = useCallback((): string => {
    const text = selectedText;
    clearSelection();
    return text;
  }, [selectedText, clearSelection]);

  return {
    selectedText,
    selectionPosition,
    getSelectedText,
    getSelectionPosition,
    clearSelection,
    consumeSelectedText,
    // This can be used to manually trigger a selection check
    updateSelection: handleSelectionChange
  };
};