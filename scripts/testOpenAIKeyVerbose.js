require('dotenv').config();
const OpenAI = require('openai');

// Get the key directly from the environment
const apiKey = process.env.OPENAI_API_KEY;
console.log('Raw key from env:', JSON.stringify(apiKey));
console.log('Key length:', apiKey?.length);
console.log('First 10 chars:', apiKey?.substring(0, 10));
console.log('Last 10 chars:', apiKey?.substring(apiKey.length - 10));

// Try to clean the key of any potential invisible characters
const cleanedKey = apiKey.replace(/\s/g, '').trim();
console.log('Cleaned key length:', cleanedKey.length);

const openai = new OpenAI({
  apiKey: cleanedKey,
});

async function testApiKey() {
  try {
    console.log('Testing OpenAI API key...');
    
    // Make a simple request to test the API key
    const completion = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo', 
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 5,
    });
    
    console.log('API key is valid!');
    console.log('Response:', completion.choices[0]?.message?.content);
  } catch (error) {
    console.error('API key is invalid:', error.message);
    console.error('Error details:', error.error);
  }
}

testApiKey();