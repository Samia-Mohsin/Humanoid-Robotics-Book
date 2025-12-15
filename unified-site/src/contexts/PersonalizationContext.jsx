import React, { createContext, useContext, useState, useEffect } from 'react';

const PersonalizationContext = createContext();

export const usePersonalization = () => {
  const context = useContext(PersonalizationContext);
  if (!context) {
    throw new Error('usePersonalization must be used within a PersonalizationProvider');
  }
  return context;
};

export const PersonalizationProvider = ({ children }) => {
  const [userPreferences, setUserPreferences] = useState({
    experienceLevel: 'intermediate',
    programmingLanguages: [],
    hardwareExperience: '',
    learningGoals: '',
    preferredTopics: []
  });

  const [personalizationEnabled, setPersonalizationEnabled] = useState(false);

  useEffect(() => {
    // Load user preferences from localStorage or API
    const savedPreferences = localStorage.getItem('user_preferences');
    if (savedPreferences) {
      setUserPreferences(JSON.parse(savedPreferences));
    }
  }, []);

  const updatePreferences = (newPreferences) => {
    setUserPreferences(prev => ({ ...prev, ...newPreferences }));
    localStorage.setItem('user_preferences', JSON.stringify({ ...userPreferences, ...newPreferences }));
  };

  const personalizeContent = (content, context = {}) => {
    if (!personalizationEnabled) return content;

    // Simple personalization logic based on user preferences
    let personalizedContent = content;

    // Adjust complexity based on experience level
    if (userPreferences.experienceLevel === 'beginner') {
      personalizedContent = content.replace(/\b(advanced|complex|sophisticated)\b/gi, 'basic');
    } else if (userPreferences.experienceLevel === 'expert') {
      personalizedContent = content.replace(/\b(basic|simple|easy)\b/gi, 'advanced');
    }

    // Add hardware-specific tips based on user's hardware experience
    if (userPreferences.hardwareExperience && userPreferences.hardwareExperience.includes('Jetson')) {
      personalizedContent += '\n\n> **Hardware Tip**: Consider implementing this on your Jetson platform for optimal performance.';
    }

    return personalizedContent;
  };

  const value = {
    userPreferences,
    updatePreferences,
    personalizationEnabled,
    setPersonalizationEnabled,
    personalizeContent
  };

  return (
    <PersonalizationContext.Provider value={value}>
      {children}
    </PersonalizationContext.Provider>
  );
};