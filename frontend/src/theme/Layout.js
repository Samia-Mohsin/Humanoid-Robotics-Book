import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import { UserProvider } from '../contexts/UserContext';
import { LanguageProvider } from '../contexts/LanguageContext';
import { TextSelectionProvider } from '../components/TextSelectionProvider';
import ChatBubble from '../components/ChatBubble';

// This component wraps the Docusaurus Layout with our contexts and chat functionality
export default function LayoutWrapper(props) {
  return (
    <UserProvider>
      <LanguageProvider>
        <TextSelectionProvider>
          <OriginalLayout {...props} />
          <ChatBubble />
        </TextSelectionProvider>
      </LanguageProvider>
    </UserProvider>
  );
}