import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { usePersonalization } from '../contexts/PersonalizationContext';
import { translateToUrdu } from '../utils/translation';

const ChapterActions = ({ content, onContentChange }) => {
  const { isAuthenticated } = useAuth();
  const { personalizationEnabled, setPersonalizationEnabled, personalizeContent } = usePersonalization();
  const [isTranslated, setIsTranslated] = useState(false);
  const [translatedContent, setTranslatedContent] = useState('');

  const handlePersonalizeToggle = () => {
    if (!isAuthenticated) {
      alert('Please sign in to use personalization features');
      return;
    }
    setPersonalizationEnabled(!personalizationEnabled);
  };

  const handleTranslateToUrdu = async () => {
    if (!isAuthenticated) {
      alert('Please sign in to use translation features');
      return;
    }

    if (!isTranslated) {
      const urduContent = await translateToUrdu(content);
      setTranslatedContent(urduContent);
      setIsTranslated(true);
      onContentChange(urduContent);
    } else {
      setIsTranslated(false);
      onContentChange(content);
    }
  };

  return (
    <div style={{ marginBottom: '20px', padding: '10px', backgroundColor: '#f8f9fa', borderRadius: '5px' }}>
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        {isAuthenticated && (
          <>
            <button
              className={`personalize-btn ${personalizationEnabled ? 'active' : ''}`}
              onClick={handlePersonalizeToggle}
              title="Adapt content to your experience level"
            >
              {personalizationEnabled ? 'Disable Personalization' : 'Personalize Content'}
            </button>

            <button
              className={`translate-btn ${isTranslated ? 'active' : ''}`}
              onClick={handleTranslateToUrdu}
              title="Translate to Urdu"
            >
              {isTranslated ? 'Show Original' : 'Translate to Urdu'}
            </button>
          </>
        )}

        {!isAuthenticated && (
          <p style={{ color: '#6c757d', fontSize: '14px' }}>
            <a href="/auth/signup" style={{ color: '#007bff' }}>Sign up</a> to unlock personalization and translation features
          </p>
        )}
      </div>
    </div>
  );
};

export default ChapterActions;