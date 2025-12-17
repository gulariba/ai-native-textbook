require('dotenv').config();
const OpenAI = require('openai');

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function testApiKey() {
  try {
    console.log('Testing OpenAI API key...');
    console.log('Key length:', process.env.OPENAI_API_KEY?.length);
    
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
  }
}

testApiKey();