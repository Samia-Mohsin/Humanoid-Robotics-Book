// frontend/src/components/ChapterHeader.tsx
import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Settings, Languages, RotateCcw, UserRound } from 'lucide-react';
import { apiClient } from '../lib/api';
import { useUser } from '../contexts/UserContext';

interface ChapterHeaderProps {
  chapterId: string;
  chapterTitle: string;
  onContentUpdate?: (content: string) => void;
}

const ChapterHeader: React.FC<ChapterHeaderProps> = ({
  chapterId,
  chapterTitle
}) => {
  const { user, isAuthenticated } = useUser();
  const [isLoading, setIsLoading] = useState({
    personalize: false,
    translate: false,
    progress: false
  });
  const [progress, setProgress] = useState(0);
  const [currentContent, setCurrentContent] = useState(chapterTitle);
  const [originalContent, setOriginalContent] = useState(chapterTitle);

  // Load chapter progress when component mounts
  useEffect(() => {
    if (isAuthenticated && user) {
      loadChapterProgress();
    }
  }, [chapterId, isAuthenticated, user]);

  const loadChapterProgress = async () => {
    if (!isAuthenticated || !user) return;

    try {
      setIsLoading(prev => ({ ...prev, progress: true }));
      const progressData = await apiClient.getChapterProgress(chapterId, user.id);
      setProgress(progressData.completionPercentage);
    } catch (error) {
      console.error('Error loading chapter progress:', error);
    } finally {
      setIsLoading(prev => ({ ...prev, progress: false }));
    }
  };

  const handlePersonalize = async () => {
    if (!isAuthenticated || !user) {
      alert('Please log in to use personalization features');
      return;
    }

    try {
      setIsLoading(prev => ({ ...prev, personalize: true }));

      const request = {
        chapterId,
        userId: user.id,
        learningStyle: user.preferences?.learningStyle,
        preferences: user.preferences
      };

      const response = await apiClient.personalize(request);
      setCurrentContent(response.personalizedContent);

      // Call parent callback to update main content
      if (onContentUpdate) {
        onContentUpdate(response.personalizedContent);
      }

      // Update progress after personalization
      await updateProgress(0); // This would be based on actual reading progress

    } catch (error) {
      console.error('Error personalizing content:', error);
      alert('Failed to personalize content. Please try again.');
    } finally {
      setIsLoading(prev => ({ ...prev, personalize: false }));
    }
  };

  const handleTranslate = async () => {
    if (!isAuthenticated || !user) {
      alert('Please log in to use translation features');
      return;
    }

    try {
      setIsLoading(prev => ({ ...prev, translate: true }));

      const request = {
        chapterId,
        userId: user.id,
        targetLanguage: 'ur' // Urdu
      };

      const response = await apiClient.translate(request);

      // Update local state for the header
      setCurrentContent(response.translatedContent);

      // Call parent callback to update main content
      if (onContentUpdate) {
        onContentUpdate(response.translatedContent);
      }

    } catch (error) {
      console.error('Error translating content:', error);
      alert('Failed to translate content. Please try again.');
    } finally {
      setIsLoading(prev => ({ ...prev, translate: false }));
    }
  };

  const handleResetContent = () => {
    setCurrentContent(originalContent);
    // Also reset the main content if there's a callback
    if (onContentUpdate) {
      onContentUpdate(originalContent);
    }
  };

  const updateProgress = async (percentage: number) => {
    if (!isAuthenticated || !user) return;

    try {
      const request = {
        completionPercentage: percentage,
        timeSpent: 0, // This would be calculated based on actual time spent
        bookmarks: [] // This would be updated based on user bookmarks
      };

      const response = await apiClient.updateChapterProgress(chapterId, request, user.id);
      setProgress(response.completionPercentage);
    } catch (error) {
      console.error('Error updating chapter progress:', error);
    }
  };

  return (
    <div className="w-full bg-white border-b border-gray-200 py-4 px-6 shadow-sm">
      <div className="max-w-8xl mx-auto flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold text-gray-900 truncate">
            {currentContent}
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Chapter {chapterId}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Progress Circle - Only show if user is authenticated */}
          {isAuthenticated && user && (
            <div className="flex items-center gap-2 min-w-fit">
              <div className="relative w-10 h-10">
                <Progress
                  value={progress}
                  className="w-10 h-10 rounded-full border-2 border-gray-200"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xs font-medium text-gray-700">
                    {Math.round(progress)}%
                  </span>
                </div>
              </div>
              <span className="text-sm text-gray-600 hidden sm:block">
                Complete
              </span>
            </div>
          )}

          {/* Personalization and Translation buttons - only for authenticated users */}
          {isAuthenticated && user ? (
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handlePersonalize}
                disabled={isLoading.personalize}
                className="flex items-center gap-2"
              >
                {isLoading.personalize ? (
                  <RotateCcw className="h-4 w-4 animate-spin" />
                ) : (
                  <Settings className="h-4 w-4" />
                )}
                <span className="hidden sm:inline">Personalize for Me</span>
                <span className="sm:hidden">Personalize</span>
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={handleTranslate}
                disabled={isLoading.translate}
                className="flex items-center gap-2"
              >
                {isLoading.translate ? (
                  <RotateCcw className="h-4 w-4 animate-spin" />
                ) : (
                  <Languages className="h-4 w-4" />
                )}
                <span className="hidden sm:inline">اردو میں ترجمہ</span>
                <span className="sm:hidden">Translate</span>
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={handleResetContent}
                disabled={currentContent === originalContent}
                className="flex items-center gap-2"
              >
                <RotateCcw className="h-4 w-4" />
                <span className="hidden sm:inline">Reset</span>
              </Button>
            </div>
          ) : (
            // Show login prompt for unauthenticated users
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <UserRound className="h-4 w-4" />
              <span>Login to personalize</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChapterHeader;