// frontend/src/lib/auth.ts
import { createAuth } from '@better-auth/react';

// Initialize Better Auth client
export const auth = createAuth({
  baseURL: process.env.REACT_APP_AUTH_BASE_URL || 'http://localhost:8000',
  // Additional configuration options can be added here
  // based on the backend auth setup
});

// Helper function to check if user is authenticated
export const isAuthenticated = (): boolean => {
  // This would typically check for a valid session token
  // Implementation depends on how the backend handles sessions
  const token = localStorage.getItem('auth_token');
  return !!token;
};

// Helper function to get user session
export const getUserSession = async () => {
  // This would typically make an API call to verify session
  // and return user information
  try {
    const token = localStorage.getItem('auth_token');
    if (!token) {
      return null;
    }

    const response = await fetch(`${process.env.REACT_APP_AUTH_BASE_URL || 'http://localhost:8000'}/api/auth/session`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      localStorage.removeItem('auth_token');
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting user session:', error);
    localStorage.removeItem('auth_token');
    return null;
  }
};

// Helper function to store authentication token
export const setAuthToken = (token: string) => {
  localStorage.setItem('auth_token', token);
};

// Helper function to clear authentication token
export const clearAuthToken = () => {
  localStorage.removeItem('auth_token');
};

// Type definitions for user session
export interface UserSession {
  id: string;
  email: string;
  name?: string;
  preferences?: {
    learningStyle?: 'visual' | 'auditory' | 'reading' | 'kinesthetic';
    languagePreferences?: string[];
    accessibilitySettings?: {
      highContrast?: boolean;
      fontSize?: 'small' | 'normal' | 'large' | 'xlarge';
      screenReader?: boolean;
      reducedMotion?: boolean;
    };
  };
  createdAt: string;
  lastLoginAt: string;
}