import React, { useState } from 'react';

interface PersonalizeContentButtonProps {
  onPersonalize: (level: string) => void;
}

const PersonalizeContentButton: React.FC<PersonalizeContentButtonProps> = ({ onPersonalize }) => {
  const [showDropdown, setShowDropdown] = useState(false);

  const handlePersonalize = (level: string) => {
    onPersonalize(level);
    setShowDropdown(false);
  };

  return (
    <div className="personalize-content-container">
      <button
        className="personalize-btn"
        onClick={() => setShowDropdown(!showDropdown)}
        title="Personalize content based on your skill level"
      >
        Personalize Content
      </button>

      {showDropdown && (
        <div className="personalize-dropdown">
          <div
            className="dropdown-item"
            onClick={() => handlePersonalize('Beginner')}
          >
            Beginner
          </div>
          <div
            className="dropdown-item"
            onClick={() => handlePersonalize('Intermediate')}
          >
            Intermediate
          </div>
          <div
            className="dropdown-item"
            onClick={() => handlePersonalize('Advanced')}
          >
            Advanced
          </div>
        </div>
      )}
    </div>
  );
};

export default PersonalizeContentButton;