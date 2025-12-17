# Data Model: AI-Native Physical & Humanoid Robotics Textbook

**Date**: 2025-12-08

## Key Entities

### User Profile
- **id**: string (unique identifier)
- **background**: string (technical background level: beginner, intermediate, advanced)
- **learningGoals**: array<string> (user's learning objectives)
- **preferredLanguage**: string (primary language preference: en, ur)
- **translationEnabled**: boolean (whether to show translations)
- **learningStyle**: string (learning style: visual, auditory, hands-on, reading)
- **technicalDepthPreference**: string (content depth preference: simplified, balanced, detailed)
- **preferences**: object (detailed content preferences)
- **learningProgress**: object (progress across different chapters/topics)
- **createdAt**: datetime
- **updatedAt**: datetime

### Textbook Chapter
- **id**: string (unique identifier)
- **title**: string (chapter title)
- **description**: string (brief description of the chapter)
- **content**: string (the actual chapter content in markdown)
- **contentType**: enum (text, video, interactive simulation, exercise-set)
- **difficultyLevel**: enum (beginner, intermediate, advanced)
- **estimatedReadingTime**: number (estimated time to complete in minutes)
- **topics**: array<string> (specific topics covered in the chapter)
- **exercises**: array<object> (practice exercises at the end of chapter)
- **examples**: array<object> (code examples or practical examples)
- **urduTranslation**: string (Urdu translation of the content)
- **urduAvailable**: boolean (indicates if Urdu translation is available)
- **personalizedContentCache**: object (cache of personalized versions)
- **createdAt**: datetime
- **updatedAt**: datetime

### RAG Knowledge Base Entry
- **id**: string (unique identifier)
- **chapterId**: string (reference to the chapter)
- **contentChunk**: string (a segment of content for RAG retrieval)
- **chunkIndex**: number (order of chunk within chapter)
- **embedding**: array<number> (vector embedding of the content chunk)
- **metadata**: object (additional metadata for retrieval including difficulty, topics)
- **language**: string (language of the content: en, ur)
- **createdAt**: datetime

### Learning Analytics
- **id**: string (unique identifier)
- **userId**: string (reference to user)
- **chapterId**: string (reference to chapter)
- **sessionId**: string (reference to the learning session)
- **timeSpent**: number (time spent on the chapter in seconds)
- **progress**: number (percentage of chapter completed: 0-100)
- **engagementMetrics**: object (metrics like clicks, page interactions, time on sections)
- **quizScores**: array<object> (scores from chapter quizzes)
- **lastAccessed**: datetime
- **completed**: boolean
- **timestamp**: datetime

### Personalization Adaptation Log
- **id**: string (unique identifier)
- **userId**: string (reference to user)
- **chapterId**: string (reference to chapter)
- **originalDifficulty**: string (difficulty level before adaptation)
- **adaptedDifficulty**: string (difficulty level after adaptation)
- **adaptationReasons**: array<string> (reasons for the adaptations made)
- **contentChanges**: array<object> (summary of content changes)
- **userFeedbackRating**: number (user rating of personalization 1-5)
- **createdAt**: datetime
- **updatedAt**: datetime

## Relationships

- User Profile `1` → `M` Learning Analytics (user has many analytics records)
- User Profile `1` → `M` Personalization Adaptation Log (user has many adaptation logs)
- Textbook Chapter `1` → `M` RAG Knowledge Base Entry (chapter has many content chunks)
- User Profile `1` → `1` User Preferences (user has one preferences object)
- Learning Analytics `1` → `M` Quiz Attempts (analytics record has many quiz attempts)