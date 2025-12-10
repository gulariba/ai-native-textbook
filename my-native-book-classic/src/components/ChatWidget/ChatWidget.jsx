import React, { useState, useRef, useEffect } from 'react';
import './chat.css';

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useSelectedText, setUseSelectedText] = useState(false);
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
    const userMessage = { type: 'user', content: inputValue };
    setMessages(prev => [...prev, userMessage]);
    const newInputValue = inputValue;
    setInputValue('');
    setIsLoading(true);

    try {
      let response;
      if (useSelectedText) {
        // Get selected text from the page
        const selectedText = window.getSelection().toString();
        
        if (!selectedText.trim()) {
          throw new Error('No text selected. Please select some text on the page first.');
        }

        // Send to ask_selected endpoint
        response = await fetch('http://localhost:8000/ask_selected', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: newInputValue,
            selected_text: selectedText
          })
        });
      } else {
        // Send to ask endpoint
        response = await fetch('http://localhost:8000/ask', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: newInputValue
          })
        });
      }

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      const botMessage = { type: 'bot', content: data.answer };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = { type: 'bot', content: `Sorry, I encountered an error: ${error.message}` };
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

  const toggleSelectedTextMode = () => {
    setUseSelectedText(!useSelectedText);
  };

  return (
    <>
      {/* Floating chat button */}
      {!isOpen && (
        <button className="chat-float-button" onClick={toggleChat}>
          💬
        </button>
      )}

      {/* Chat widget */}
      {isOpen && (
        <div className="chat-widget">
          <div className="chat-header">
            <h3>Book Assistant</h3>
            <div className="header-buttons">
              <button 
                className={`use-selected-btn ${useSelectedText ? 'active' : ''}`} 
                onClick={toggleSelectedTextMode}
                title="Toggle between using entire book and selected text only"
              >
                {useSelectedText ? 'Using Selection' : 'Use Entire Book'}
              </button>
              <button className="close-button" onClick={toggleChat}>×</button>
            </div>
          </div>
          
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Hello! I'm your book assistant. Ask me questions about the content.</p>
                <p>{useSelectedText 
                  ? 'Currently using selected text only.' 
                  : 'Currently using the entire book.'}</p>
              </div>
            ) : (
              messages.map((msg, index) => (
                <div key={index} className={`message ${msg.type}`}>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message bot">
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
          
          <div className="chat-input-area">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={useSelectedText 
                ? "Ask about the selected text..." 
                : "Ask about the book..."}
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