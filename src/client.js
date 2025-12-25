// Client-side entry point to set up personalization context
import { PersonalizationProvider } from './components/PersonalizationProvider';

export const wrapRootElement = ({ element }) => {
  return (
    <PersonalizationProvider>
      {element}
    </PersonalizationProvider>
  );
};