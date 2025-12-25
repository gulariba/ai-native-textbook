import React from 'react';
import { usePersonalization } from './PersonalizationProvider';

interface ContentPersonalizerProps {
  beginnerContent: React.ReactNode;
  intermediateContent: React.ReactNode;
  advancedContent: React.ReactNode;
  className?: string;
}

const ContentPersonalizer: React.FC<ContentPersonalizerProps> = ({
  beginnerContent,
  intermediateContent,
  advancedContent,
  className = ''
}) => {
  const { contentLevel } = usePersonalization();

  const renderContent = () => {
    switch(contentLevel) {
      case 'Intermediate':
        return intermediateContent;
      case 'Advanced':
        return advancedContent;
      case 'Beginner':
      default:
        return beginnerContent;
    }
  };

  return <div className={className}>{renderContent()}</div>;
};

export default ContentPersonalizer;