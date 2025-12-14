// frontend/src/components/ChatBubble.tsx
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bot, X, MessageCircle } from 'lucide-react';
import ChatBot from './ChatBot';

interface ChatBubbleProps {
  onContextChange?: (context: string) => void;
}

const ChatBubble: React.FC<ChatBubbleProps> = ({ onContextChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [hasUnreadMessage, setHasUnreadMessage] = useState(false);

  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setHasUnreadMessage(false);
    }
  };

  // This would be called when a new message arrives
  const onNewMessage = () => {
    if (!isOpen) {
      setHasUnreadMessage(true);
    }
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <AnimatePresence>
        {isOpen ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 20 }}
            transition={{ duration: 0.2 }}
            className="mb-4"
          >
            <ChatBot
              onClose={toggleChat}
              onNewMessage={onNewMessage}
              onContextChange={onContextChange}
            />
          </motion.div>
        ) : null}
      </AnimatePresence>

      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        onClick={toggleChat}
        aria-label={isOpen ? "Close chat" : "Open chat"}
        className={`
          w-14 h-14 rounded-full flex items-center justify-center shadow-lg
          bg-gradient-to-r from-blue-500 to-purple-600 text-white
          hover:from-blue-600 hover:to-purple-700
          focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75
          transition-all duration-200
          ${hasUnreadMessage ? 'animate-pulse ring-4 ring-blue-400 ring-opacity-50' : ''}
        `}
      >
        {isOpen ? (
          <X className="w-6 h-6" />
        ) : (
          <>
            <Bot className="w-6 h-6" />
            {hasUnreadMessage && (
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center text-xs text-white">
                !
              </span>
            )}
          </>
        )}
      </motion.button>
    </div>
  );
};

export default ChatBubble;