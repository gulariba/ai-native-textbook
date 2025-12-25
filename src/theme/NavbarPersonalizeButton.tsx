import React from 'react';
import { usePersonalization } from '../components/PersonalizationProvider';

const NavbarPersonalizeButton = () => {
  const { contentLevel, setPersonalizationLevel } = usePersonalization();

  const togglePersonalization = () => {
    const newLevel = contentLevel === 'Beginner' ? 'Advanced' : 'Beginner';
    setPersonalizationLevel(newLevel);
    // Store in localStorage so it persists across page navigation
    localStorage.setItem('contentLevel', newLevel);
  };

  return (
    <div className="navbar__item">
      <button
        className="personalize-toggle-btn"
        onClick={togglePersonalization}
        title={`Switch to ${contentLevel === 'Beginner' ? 'Advanced' : 'Beginner'} content`}
      >
        {contentLevel} Mode
      </button>
    </div>
  );
};

export default NavbarPersonalizeButton;