import React, { useState, useEffect } from 'react';
import OriginalDocItem from '@theme-original/DocItem';
import ChapterHeader from '../components/ChapterHeader';
import { useLocation } from '@docusaurus/router';

// This component wraps the Docusaurus DocItem with our ChapterHeader
export default function DocItemWrapper(props) {
  const { route } = props;
  const location = useLocation();
  const [translatedContent, setTranslatedContent] = useState(null);

  // Extract chapter ID and title from the route
  let chapterId = '';
  let chapterTitle = '';

  if (route) {
    // Get the last part of the pathname as the chapter ID
    chapterId = location.pathname.split('/').pop() || 'unknown';

    // Try to get the title from the route
    chapterTitle = route.title || chapterId;
  }

  // Fallback to document title if available in props
  if (!chapterTitle && props.content) {
    const { metadata } = props.content;
    if (metadata) {
      chapterTitle = metadata.title || metadata.unversionedId || chapterId;
    }
  }

  // Function to handle translation updates from ChapterHeader
  const handleContentUpdate = (newContent) => {
    setTranslatedContent(newContent);
  };

  // Reset translated content when location changes
  useEffect(() => {
    setTranslatedContent(null);
  }, [location.pathname]);

  // Create a context for passing the content update function to ChapterHeader
  const DocItemWithTranslation = () => {
    // We'll need to pass the content update function to the ChapterHeader
    // For now, we'll just render the header with a custom callback
    return (
      <>
        <ChapterHeader
          chapterId={chapterId}
          chapterTitle={chapterTitle}
          onContentUpdate={handleContentUpdate}
        />
        {translatedContent ? (
          <div className="container margin-vert--lg">
            <div className="row">
              <div className="col col--8 col--offset-2">
                <article>
                  <header>
                    <h1 className="docTitle">{chapterTitle}</h1>
                  </header>
                  <div className="markdown">
                    {translatedContent}
                  </div>
                </article>
              </div>
            </div>
          </div>
        ) : (
          <OriginalDocItem {...props} />
        )}
      </>
    );
  };

  return <DocItemWithTranslation />;
}