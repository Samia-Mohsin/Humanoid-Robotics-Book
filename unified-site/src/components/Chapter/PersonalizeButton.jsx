import React, { useState, useContext } from 'react';
import { AuthContext } from '../../contexts/AuthContext';
import { PersonalizationContext } from '../../contexts/PersonalizationContext';

const PersonalizeButton = ({ content, variant = 'full', onPersonalize }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const { user } = useContext(AuthContext);
  const { userPreferences } = useContext(PersonalizationContext);

  const handlePersonalize = async () => {
    if (!user) {
      alert('Please sign in to use personalization feature');
      return;
    }

    setIsProcessing(true);
    try {
      const response = await fetch('/api/content/personalize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({
          content: content,
          context: userPreferences || {}
        })
      });

      if (!response.ok) {
        throw new Error('Personalization failed');
      }

      const data = await response.json();

      if (onPersonalize) {
        onPersonalize(data.personalized_content);
      }

      if (variant === 'full') {
        alert('Content has been personalized based on your preferences!');
      }
    } catch (error) {
      console.error('Personalization error:', error);
      alert('Personalization failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  if (variant === 'compact') {
    return (
      <button
        onClick={handlePersonalize}
        disabled={isProcessing}
        className="w-full bg-blue-100 hover:bg-blue-200 text-blue-800 dark:bg-blue-900 dark:hover:bg-blue-800 dark:text-blue-200 px-3 py-1 rounded text-sm transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
      >
        {isProcessing ? (
          <span className="flex items-center">
            <span className="h-3 w-3 bg-blue-600 rounded-full mr-1 animate-bounce"></span>
            <span className="h-3 w-3 bg-blue-600 rounded-full mr-1 animate-bounce" style={{ animationDelay: '0.2s' }}></span>
            <span className="h-3 w-3 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
          </span>
        ) : (
          <span>Personalize for Me</span>
        )}
      </button>
    );
  }

  return (
    <button
      onClick={handlePersonalize}
      disabled={isProcessing}
      className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
    >
      {isProcessing ? (
        <span className="flex items-center">
          <span className="h-3 w-3 bg-white rounded-full mr-2 animate-bounce"></span>
          <span className="h-3 w-3 bg-white rounded-full mr-2 animate-bounce" style={{ animationDelay: '0.2s' }}></span>
          <span className="h-3 w-3 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
          <span>Processing...</span>
        </span>
      ) : (
        <span>Personalize for Me</span>
      )}
    </button>
  );
};

export default PersonalizeButton;