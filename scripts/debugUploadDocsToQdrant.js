const fs = require('fs');
const path = require('path');
require('dotenv').config();

const { QdrantClient } = require('@qdrant/js-client-rest');
const OpenAI = require('openai');

console.log('QDRANT_URL:', process.env.QDRANT_URL ? 'SET' : 'NOT SET');
console.log('QDRANT_API_KEY:', process.env.QDRANT_API_KEY ? 'SET' : 'NOT SET');
console.log('OPENAI_API_KEY length:', process.env.OPENAI_API_KEY ? process.env.OPENAI_API_KEY.length : 'NOT SET');

// Initialize Qdrant client
const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Collection name for the textbook
const COLLECTION_NAME = 'physical_ai_robotics_book';

// Function to clean markdown content
function cleanMarkdown(content) {
  // Remove markdown headers, links, images, etc.
  return content
    .replace(/#{1,6}\s/g, '') // Headers
    .replace(/\*\*(.*?)\*\*/g, '$1') // Bold
    .replace(/\*(.*?)\*/g, '$1') // Italic
    .replace(/\[(.*?)\]\(.*?\)/g, '$1') // Links
    .replace(/!\[.*?\]\(.*?\)/g, '') // Images
    .replace(/`{1,3}[\s\S]*?`{1,3}/g, '') // Code blocks
    .replace(/`[^`]+`/g, '') // Inline code
    .replace(/\n{3,}/g, '\n\n') // Multiple newlines to double
    .trim();
}

// Function to chunk content into segments of n words
function chunkContent(content, maxWords = 400) {
  const paragraphs = content.split('\n\n');
  const chunks = [];
  let currentChunk = '';
  
  for (const paragraph of paragraphs) {
    const paragraphWordCount = paragraph.trim().split(/\s+/).length;
    const currentWordCount = currentChunk.trim().split(/\s+/).length;
    
    if (currentWordCount + paragraphWordCount <= maxWords) {
      currentChunk += paragraph + '\n\n';
    } else {
      if (currentChunk.trim()) {
        chunks.push(currentChunk.trim());
      }
      
      if (paragraphWordCount > maxWords) {
        // If a single paragraph exceeds maxWords, split it
        const words = paragraph.trim().split(/\s+/);
        currentChunk = '';
        
        for (const word of words) {
          const tempChunk = currentChunk ? `${currentChunk} ${word}` : word;
          if (tempChunk.split(/\s+/).length <= maxWords) {
            currentChunk = tempChunk;
          } else {
            if (currentChunk.trim()) {
              chunks.push(currentChunk.trim());
            }
            currentChunk = word;
          }
        }
      } else {
        currentChunk = paragraph + '\n\n';
      }
    }
  }
  
  if (currentChunk.trim()) {
    chunks.push(currentChunk.trim());
  }
  
  return chunks;
}

// Function to read all md/mdx files from docs directory
async function readDocsFromDirectory(dirPath) {
  const files = fs.readdirSync(dirPath);
  const docs = [];
  
  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory()) {
      // Recursively read subdirectories
      const subDocs = await readDocsFromDirectory(filePath);
      docs.push(...subDocs);
    } else if (file.endsWith('.md') || file.endsWith('.mdx')) {
      const content = fs.readFileSync(filePath, 'utf8');
      docs.push({
        filename: file,
        filepath: filePath,
        content: content,
        title: path.basename(file, path.extname(file)),
      });
    }
  }
  
  return docs;
}

// Function to generate embedding for text
async function generateEmbedding(text) {
  try {
    console.log('Generating embedding for text:', text.substring(0, 50) + '...');
    const response = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: text,
    });
    
    return response.data[0].embedding;
  } catch (error) {
    console.error('Error generating embedding:', error.message);
    throw error;
  }
}

// Main function to upload docs to Qdrant
async function uploadDocsToQdrant() {
  try {
    console.log(`Starting document upload to Qdrant collection: ${COLLECTION_NAME}`);
    
    // Read all docs from the docs directory
    const docs = await readDocsFromDirectory('./docs');
    console.log(`Found ${docs.length} documents`);
    
    // Check if collection exists, create if it doesn't
    try {
      await qdrantClient.getCollection(COLLECTION_NAME);
      console.log(`Collection ${COLLECTION_NAME} already exists, reusing it.`);
    } catch (error) {
      console.log(`Collection ${COLLECTION_NAME} does not exist, creating it...`);
      await qdrantClient.createCollection(COLLECTION_NAME, {
        vector_size: 1536, // Size of OpenAI embeddings
        distance: 'Cosine',
      });
      console.log(`Collection ${COLLECTION_NAME} created successfully.`);
    }

    // Prepare points for insertion
    const points = [];
    let pointId = 0;
    
    for (const doc of docs) {
      const cleanedContent = cleanMarkdown(doc.content);
      const chunks = chunkContent(cleanedContent, 400); // About 400 words per chunk
      
      for (const chunk of chunks) {
        console.log(`Processing chunk from ${doc.filename}: ${chunk.substring(0, 100)}...`);
        
        try {
          const embedding = await generateEmbedding(chunk);
          console.log(`Generated embedding with ${embedding.length} dimensions`);
          
          points.push({
            id: pointId++,
            vector: embedding,
            payload: {
              content: chunk,
              title: doc.title,
              filename: doc.filename,
              filepath: doc.filepath,
            },
          });
        } catch (embeddingError) {
          console.error(`Error processing chunk from ${doc.filename}:`, embeddingError.message);
          continue; // Skip this chunk and continue with the next one
        }
      }
    }
    
    console.log(`Uploading ${points.length} vectors to Qdrant...`);
    
    // Batch upload to Qdrant
    const batchSize = 100;
    for (let i = 0; i < points.length; i += batchSize) {
      const batch = points.slice(i, i + batchSize);
      await qdrantClient.upsert(COLLECTION_NAME, {
        wait: true,
        points: batch,
      });
      console.log(`Uploaded batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(points.length / batchSize)}`);
    }
    
    console.log(`Successfully uploaded ${points.length} vectors to Qdrant collection: ${COLLECTION_NAME}`);
  } catch (error) {
    console.error('Error uploading docs to Qdrant:', error);
  }
}

// Run the upload function
uploadDocsToQdrant()
  .then(() => console.log('Upload completed'))
  .catch(console.error);