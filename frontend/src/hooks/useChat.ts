// frontend/src/hooks/useChat.ts
import { useState, useEffect, useCallback } from 'react';
import { apiClient, ChatRequest, ChatResponse } from '../lib/api';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  context?: string;
}

export interface ChatSession {
  id: string;
  messages: Message[];
  createdAt: Date;
  lastActive: Date;
  context?: string;
}

export const useChat = () => {
  const [sessionId, setSessionId] = useState<string>(() => {
    // Generate or retrieve session ID from localStorage
    const savedSessionId = localStorage.getItem('chat_session_id');
    if (savedSessionId) {
      return savedSessionId;
    }
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('chat_session_id', newSessionId);
    return newSessionId;
  });

  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [context, setContext] = useState<string>('');

  // Load messages from localStorage if available
  useEffect(() => {
    const savedMessages = localStorage.getItem(`chat_messages_${sessionId}`);
    if (savedMessages) {
      try {
        const parsedMessages = JSON.parse(savedMessages);
        setMessages(parsedMessages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        })));
      } catch (e) {
        console.error('Error loading chat messages from localStorage:', e);
      }
    }
  }, [sessionId]);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem(`chat_messages_${sessionId}`, JSON.stringify(messages));
  }, [messages, sessionId]);

  const sendMessage = useCallback(async (message: string, additionalContext?: string) => {
    try {
      setIsLoading(true);
      setError(null);

      // Add user message to the conversation
      const userMessage: Message = {
        id: `msg_${Date.now()}`,
        role: 'user',
        content: message,
        timestamp: new Date(),
        context: additionalContext || context
      };

      setMessages(prev => [...prev, userMessage]);

      // Prepare the request to the API
      const request: ChatRequest = {
        message,
        context: additionalContext || context,
        sessionId,
      };

      // Get user ID if available
      const userId = localStorage.getItem('user_id');
      if (userId) {
        request.userId = userId;
      }

      // Clear context after using it
      setContext('');

      // Call the API
      const response = await apiClient.chat(request);

      // Add assistant response to the conversation
      const assistantMessage: Message = {
        id: response.id,
        role: 'assistant',
        content: response.response,
        timestamp: new Date(response.timestamp),
        context: response.contextUsed ? (additionalContext || context) : undefined
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error sending message:', err);
      setError(err instanceof Error ? err.message : 'Failed to send message');

      // Add error message to the conversation
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, context]);

  const clearChat = useCallback(() => {
    setMessages([]);
    // Generate a new session ID when clearing chat
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    localStorage.setItem('chat_session_id', newSessionId);
    localStorage.removeItem(`chat_messages_${sessionId}`);
  }, [sessionId]);

  const setSelectionContext = useCallback((selectedText: string) => {
    // Limit context to 1000 characters to prevent overly long contexts
    const limitedContext = selectedText.length > 1000
      ? selectedText.substring(0, 1000) + '...'
      : selectedText;
    setContext(limitedContext);
  }, []);

  return {
    messages,
    sendMessage,
    isLoading,
    error,
    clearChat,
    context,
    setSelectionContext,
    sessionId
  };
};