import { useState, useEffect } from 'react';

const useTextSelection = () => {
  const [selectedText, setSelectedText] = useState('');

  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      const text = selection.toString().trim();

      if (text) {
        setSelectedText(text);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', (e) => {
      if (e.key === 'Escape') {
        setSelectedText('');
      }
    });

    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', (e) => {
        if (e.key === 'Escape') {
          setSelectedText('');
        }
      });
    };
  }, []);

  const clearSelection = () => {
    setSelectedText('');
    window.getSelection().removeAllRanges();
  };

  return { selectedText, clearSelection };
};

export default useTextSelection;