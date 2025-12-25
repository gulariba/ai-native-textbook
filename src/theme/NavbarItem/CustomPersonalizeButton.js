import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import { NavbarNavLink } from '@docusaurus/theme-common';

const CustomNavbarPersonalizeButton = (props) => {
  const [contentLevel, setContentLevel] = useState('Beginner');

  // Use localStorage to manage the content level
  
  const handleLevelChange = (level) => {
    setContentLevel(level);
    localStorage.setItem('contentLevel', level);
    // Dispatch a custom event to trigger re-render of content
    window.dispatchEvent(new CustomEvent('contentLevelChanged', { detail: { level } }));
  };

  return (
    <div className="navbar__item dropdown dropdown--hoverable dropdown--right">
      <span className="navbar__link">
        Personalize ({contentLevel})
      </span>
      <ul className="dropdown__menu">
        <li>
          <a 
            href="#" 
            className="dropdown__link"
            onClick={(e) => { e.preventDefault(); handleLevelChange('Beginner'); }}
          >
            Beginner
          </a>
        </li>
        <li>
          <a 
            href="#" 
            className="dropdown__link"
            onClick={(e) => { e.preventDefault(); handleLevelChange('Intermediate'); }}
          >
            Intermediate
          </a>
        </li>
        <li>
          <a 
            href="#" 
            className="dropdown__link"
            onClick={(e) => { e.preventDefault(); handleLevelChange('Advanced'); }}
          >
            Advanced
          </a>
        </li>
      </ul>
    </div>
  );
};

export default CustomNavbarPersonalizeButton;