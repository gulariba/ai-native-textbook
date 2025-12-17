import React, { useState, useRef, useEffect } from 'react';
import './chat.css';

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sources, setSources] = useState([]);
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = { role: 'user', content: inputValue };
    setMessages(prev => [...prev, userMessage]);
    const newInputValue = inputValue;
    setInputValue('');
    setIsLoading(true);

    try {
      // Send to our RAG API route
      const response = await fetch('/api/chat/route', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...messages, userMessage]
        })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      const botMessage = { role: 'assistant', content: data.message };
      setMessages(prev => [...prev, botMessage]);
      setSources(data.sources || []);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = { role: 'assistant', content: `Sorry, I encountered an error: ${error.message}` };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Floating chat button */}
      {!isOpen && (
        <button className="chat-float-button" onClick={toggleChat}>
          Ask the Book 🤖
        </button>
      )}

      {/* Chat widget */}
      {isOpen && (
        <div className="chat-widget">
          <div className="chat-header">
            <h3>Ask the Book 🤖</h3>
            <button className="close-button" onClick={toggleChat}>×</button>
          </div>

          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Hello! I'm your AI assistant for the Physical AI & Humanoid Robotics textbook.</p>
                <p>Ask me anything about the content!</p>
              </div>
            ) : (
              messages.map((msg, index) => (
                <div key={index} className={`message ${msg.role}`}>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message assistant">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Sources section */}
          {sources.length > 0 && (
            <div className="sources-section">
              <h4>Sources:</h4>
              <ul>
                {sources.map((source, idx) => (
                  <li key={idx} className="source-item">
                    <strong>{source.title}</strong>: {source.content}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="chat-input-area">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about Physical AI & Humanoid Robotics..."
              disabled={isLoading}
              rows="2"
            />
            <button
              onClick={handleSend}
              disabled={!inputValue.trim() || isLoading}
              className="send-button"
            >
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatWidget;