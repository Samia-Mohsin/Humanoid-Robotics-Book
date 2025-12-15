import React from 'react';
import { AuthProvider } from './contexts/AuthContext';
import { PersonalizationProvider } from './contexts/PersonalizationContext';

export default function Root({ children }) {
  return (
    <AuthProvider>
      <PersonalizationProvider>
        {children}
      </PersonalizationProvider>
    </AuthProvider>
  );
}