import React, { useState, useEffect, useRef } from 'react';
import ChatInterface from './ChatInterface';
import { FaComment, FaTimes } from 'react-icons/fa';

const FloatingChat = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [isMinimized, setIsMinimized] = useState(false);
  const chatRef = useRef(null);

  // Function to capture selected text
  useEffect(() => {
    const handleTextSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleTextSelection);
    return () => {
      document.removeEventListener('mouseup', handleTextSelection);
    };
  }, []);

  // Close chat when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (chatRef.current && !chatRef.current.contains(event.target) && isOpen) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const toggleChat = () => {
    if (!isOpen) {
      setIsOpen(true);
      setIsMinimized(false);
    } else {
      setIsMinimized(!isMinimized);
    }
  };

  const closeChat = () => {
    setIsOpen(false);
    setIsMinimized(false);
  };

  return (
    <div className="fixed z-50 bottom-6 right-6">
      {isOpen && !isMinimized ? (
        <div
          ref={chatRef}
          className="bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 w-80 h-96 flex flex-col"
        >
          <div className="flex justify-between items-center bg-blue-600 dark:bg-blue-700 text-white p-3 rounded-t-lg">
            <h3 className="font-semibold">AI Assistant</h3>
            <div className="flex space-x-2">
              <button
                onClick={() => setIsMinimized(true)}
                className="text-white hover:text-gray-200"
              >
                _
              </button>
              <button
                onClick={closeChat}
                className="text-white hover:text-gray-200"
              >
                <FaTimes />
              </button>
            </div>
          </div>
          <ChatInterface
            selectedText={selectedText}
            onTextSelected={setSelectedText}
            className="flex-grow"
          />
        </div>
      ) : (
        <div className="relative">
          <button
            onClick={toggleChat}
            className="bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 flex items-center justify-center"
            aria-label="Open chat"
          >
            <FaComment className="text-xl" />
            {selectedText && (
              <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                !
              </span>
            )}
          </button>
        </div>
      )}
    </div>
  );
};

export default FloatingChat;