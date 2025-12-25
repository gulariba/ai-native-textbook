import React, { useState, useEffect } from 'react';
import { translate } from '@docusaurus/Translate';

const LanguageSwitcher = () => {
  const [currentLang, setCurrentLang] = useState('en');

  useEffect(() => {
    // Check current language from URL
    const pathLang = window.location.pathname.split('/')[1];
    if (pathLang === 'ur') {
      setCurrentLang('ur');
    } else {
      setCurrentLang('en');
    }
  }, []);

  const handleLanguageChange = (lang) => {
    setCurrentLang(lang);
    
    // Update the URL to switch language
    const pathSegments = window.location.pathname.split('/');
    if (lang === 'ur') {
      // If current language is English and we're switching to Urdu
      if (!window.location.pathname.startsWith('/ur/')) {
        // Add 'ur' to the path
        window.location.pathname = `/ur${window.location.pathname}`;
      }
    } else {
      // If current language is Urdu and we're switching to English
      if (window.location.pathname.startsWith('/ur/')) {
        // Remove 'ur' from the path
        window.location.pathname = window.location.pathname.replace('/ur', '');
      }
    }
  };

  return (
    <div className="language-switcher">
      <select 
        value={currentLang} 
        onChange={(e) => handleLanguageChange(e.target.value)}
        style={{
          background: 'rgba(255, 255, 255, 0.1)',
          color: 'white',
          border: '1px solid rgba(108, 99, 255, 0.3)',
          borderRadius: '60px',
          padding: '8px 16px',
          fontFamily: 'inherit',
          fontSize: 'inherit',
          cursor: 'pointer'
        }}
      >
        <option value="en">English</option>
        <option value="ur">اردو</option>
      </select>
    </div>
  );
};

export default LanguageSwitcher;