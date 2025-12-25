import React from 'react';
import PersonalizeContentButton from '@site/src/components/PersonalizeContentButton';
import { usePersonalization } from '@site/src/components/PersonalizationProvider';

const PersonalizeButton = () => {
  const { contentLevel, setPersonalizationLevel } = usePersonalization();

  const handlePersonalize = (level: string) => {
    setPersonalizationLevel(level);
    // Store in localStorage so it persists across page navigation
    localStorage.setItem('contentLevel', level);
  };

  return (
    <div className="navbar__item">
      <PersonalizeContentButton onPersonalize={handlePersonalize} />
    </div>
  );
};

export default PersonalizeButton;