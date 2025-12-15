import React, { useState, useEffect, useRef } from 'react';
import { FiSend, FiMessageSquare, FiUser, FiX, FiMinimize2, FiMaximize2 } from 'react-icons/fi';

const Chatbot = ({ isEmbedded = false }) => {
  const [isOpen, setIsOpen] = useState(!isEmbedded);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);

  // Get selected text from the page
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText.length > 0) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputMessage.trim() && !selectedText) return;

    const messageToSend = selectedText ? `${selectedText}\n\nQuestion: ${inputMessage}` : inputMessage;

    const userMessage = { id: Date.now(), text: messageToSend, sender: 'user', timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setSelectedText('');
    setIsLoading(true);

    try {
      // In a real implementation, this would call your FastAPI backend
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageToSend,
          user_id: localStorage.getItem('user_id') || 'anonymous',
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage = { id: Date.now() + 1, text: data.response, sender: 'bot', timestamp: new Date() };
        setMessages(prev => [...prev, botMessage]);
      } else {
        const botMessage = {
          id: Date.now() + 1,
          text: 'Sorry, I encountered an error. Please try again.',
          sender: 'bot',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      }
    } catch (error) {
      const botMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I\'m having trouble connecting. Please check your connection.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  if (isEmbedded) {
    return (
      <div className="embedded-chatbot">
        <div className="chat-header">
          <h3><FiMessageSquare /> AI Assistant</h3>
        </div>
        <div className="chat-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.sender}`}>
              <div className="message-content">
                {msg.sender === 'bot' ? <FiUser /> : <FiUser />}
                <div className="message-text">{msg.text}</div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="message-content">
                <FiUser />
                <div className="typing-indicator">AI is thinking...</div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="chat-input">
          {selectedText && (
            <div className="selected-text-preview">
              Selected: "{selectedText.substring(0, 50)}..."
            </div>
          )}
          <div className="input-container">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about this content..."
              rows="2"
            />
            <button onClick={sendMessage} disabled={isLoading}>
              <FiSend />
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`chatbot-container ${isOpen ? 'open' : 'closed'}`}>
      {isOpen ? (
        <div className="chatbot-window">
          <div className="chat-header">
            <h3><FiMessageSquare /> AI Assistant</h3>
            <div className="header-actions">
              <button onClick={toggleChat} className="minimize-btn">
                <FiMinimize2 />
              </button>
            </div>
          </div>
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <h4>Hello! I'm your AI assistant for Physical AI & Humanoid Robotics.</h4>
                <p>You can ask me questions about the book content or select text on the page to ask specific questions about it.</p>
              </div>
            ) : (
              messages.map((msg) => (
                <div key={msg.id} className={`message ${msg.sender}`}>
                  <div className="message-content">
                    {msg.sender === 'bot' ? <FiUser className="bot-icon" /> : <FiUser className="user-icon" />}
                    <div className="message-text">{msg.text}</div>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message bot">
                <div className="message-content">
                  <FiUser className="bot-icon" />
                  <div className="typing-indicator">AI is thinking...</div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="chat-input">
            {selectedText && (
              <div className="selected-text-preview">
                Selected: "{selectedText.substring(0, 50)}..."
              </div>
            )}
            <div className="input-container">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about the content..."
                rows="2"
                disabled={isLoading}
              />
              <button onClick={sendMessage} disabled={isLoading || (!inputMessage.trim() && !selectedText)}>
                <FiSend />
              </button>
            </div>
          </div>
        </div>
      ) : (
        <button className="chatbot-toggle" onClick={toggleChat}>
          <FiMessageSquare />
        </button>
      )}
    </div>
  );
};

export default Chatbot;