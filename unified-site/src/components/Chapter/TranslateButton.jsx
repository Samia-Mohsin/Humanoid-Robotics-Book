import React, { useState, useContext } from 'react';
import { AuthContext } from '../../contexts/AuthContext';

const TranslateButton = ({ content, variant = 'full', onTranslate }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const { user } = useContext(AuthContext);

  const handleTranslate = async () => {
    if (!user) {
      alert('Please sign in to use translation feature');
      return;
    }

    setIsProcessing(true);
    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({
          text: content,
          target_lang: 'ur',
          source_lang: 'en'
        })
      });

      if (!response.ok) {
        throw new Error('Translation failed');
      }

      const data = await response.json();

      if (onTranslate) {
        onTranslate(data.translated_text);
      }

      if (variant === 'full') {
        alert('Content has been translated to Urdu!');
      }
    } catch (error) {
      console.error('Translation error:', error);
      alert('Translation failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  if (variant === 'compact') {
    return (
      <button
        onClick={handleTranslate}
        disabled={isProcessing}
        className="w-full bg-green-100 hover:bg-green-200 text-green-800 dark:bg-green-900 dark:hover:bg-green-800 dark:text-green-200 px-3 py-1 rounded text-sm transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
      >
        {isProcessing ? (
          <span className="flex items-center">
            <span className="h-3 w-3 bg-green-600 rounded-full mr-1 animate-bounce"></span>
            <span className="h-3 w-3 bg-green-600 rounded-full mr-1 animate-bounce" style={{ animationDelay: '0.2s' }}></span>
            <span className="h-3 w-3 bg-green-600 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
          </span>
        ) : (
          <span>اردو میں ترجمہ</span>
        )}
      </button>
    );
  }

  return (
    <button
      onClick={handleTranslate}
      disabled={isProcessing}
      className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
    >
      {isProcessing ? (
        <span className="flex items-center">
          <span className="h-3 w-3 bg-white rounded-full mr-2 animate-bounce"></span>
          <span className="h-3 w-3 bg-white rounded-full mr-2 animate-bounce" style={{ animationDelay: '0.2s' }}></span>
          <span className="h-3 w-3 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
          <span>Translating...</span>
        </span>
      ) : (
        <span>اردو میں ترجمہ</span>
      )}
    </button>
  );
};

export default TranslateButton;