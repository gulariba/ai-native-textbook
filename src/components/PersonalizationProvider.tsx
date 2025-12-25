import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';

interface PersonalizationContextType {
  contentLevel: string;
  setPersonalizationLevel: (level: string) => void;
}

const PersonalizationContext = createContext<PersonalizationContextType | undefined>(undefined);

export const PersonalizationProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [contentLevel, setContentLevel] = useState<string>('Beginner'); // Default to Beginner

  useEffect(() => {
    // Set the initial content level based on localStorage
    const savedLevel = localStorage.getItem('contentLevel');
    if (savedLevel) {
      setContentLevel(savedLevel);
    }
  }, []);

  const setPersonalizationLevel = (level: string) => {
    setContentLevel(level);
    localStorage.setItem('contentLevel', level);
  };

  return (
    <PersonalizationContext.Provider value={{ contentLevel, setPersonalizationLevel }}>
      {children}
    </PersonalizationContext.Provider>
  );
};

export const usePersonalization = (): PersonalizationContextType => {
  const context = useContext(PersonalizationContext);
  if (context === undefined) {
    // Fallback to default values when not wrapped in provider
    // This handles server-side rendering and cases where provider is not available
    console.warn('usePersonalization is being used without a PersonalizationProvider. Using default values.');
    return {
      contentLevel: 'Beginner',
      setPersonalizationLevel: () => {
        // Do nothing
      }
    };
  }
  return context;
};