// pages/api/chat/route.js
import axios from 'axios';

// Backend API URL - defaults to localhost if not specified in environment
const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:8000';

export async function POST(request) {
  try {
    const { messages } = await request.json();
    
    // Extract the last user message
    const lastMessage = messages[messages.length - 1].content;
    
    // Prepare chat history (excluding the last message which is the current query)
    const chatHistory = messages.slice(0, -1).map(msg => ({
      role: msg.role,
      content: msg.content
    }));
    
    // Call the FastAPI backend
    const response = await axios.post(`${BACKEND_API_URL}/chat`, {
      message: lastMessage,
      chat_history: chatHistory
    }, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    const { response: botResponse, sources } = response.data;
    
    // Return the response from the backend
    return new Response(
      JSON.stringify({
        message: botResponse,
        sources: sources
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  } catch (error) {
    console.error('Error in chat API:', error);
    
    // Handle different types of errors
    if (error.response) {
      // Server responded with error status
      return new Response(
        JSON.stringify({
          message: 'An error occurred while processing your request.',
          error: error.response.data.detail || error.message
        }),
        {
          status: error.response.status || 500,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    } else if (error.request) {
      // Request was made but no response received
      return new Response(
        JSON.stringify({
          message: 'Could not connect to the chatbot backend.',
          error: 'Backend service is not available'
        }),
        {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    } else {
      // Something else happened
      return new Response(
        JSON.stringify({
          message: 'An error occurred while processing your request.',
          error: error.message
        }),
        {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }
  }
}