import React, { useState, useRef, useEffect } from 'react';
import clsx from 'clsx';

import styles from './ChatBot.module.css';

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sources, setSources] = useState([]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { role: 'user', content: inputValue };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat/route', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ messages: newMessages }),
      });

      const data = await response.json();
      
      const botMessage = { role: 'assistant', content: data.message };
      setMessages(prev => [...prev, botMessage]);
      setSources(data.sources || []);
    } catch (error) {
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('Chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const closeChat = () => {
    setIsOpen(false);
  };

  return (
    <>
      {/* Floating button */}
      {!isOpen && (
        <button className={styles.chatToggleButton} onClick={toggleChat}>
          Ask the Book 🤖
        </button>
      )}

      {/* Chat interface */}
      {isOpen && (
        <div className={styles.chatContainer}>
          <div className={styles.chatHeader}>
            <span className={styles.chatTitle}>Ask the Book 🤖</span>
            <button className={styles.closeButton} onClick={closeChat}>
              ×
            </button>
          </div>
          
          <div className={styles.chatMessages}>
            {messages.length === 0 ? (
              <div className={styles.welcomeMessage}>
                <p>Hello! I'm your AI assistant for the Physical AI & Humanoid Robotics textbook.</p>
                <p>Ask me anything about the content!</p>
              </div>
            ) : (
              messages.map((msg, index) => (
                <div 
                  key={index} 
                  className={clsx(styles.message, styles[msg.role])}
                >
                  <div className={styles.messageContent}>{msg.content}</div>
                </div>
              ))
            )}
            
            {isLoading && (
              <div className={clsx(styles.message, styles.assistant)}>
                <div className={styles.typingIndicator}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {sources.length > 0 && (
            <div className={styles.sourcesSection}>
              <h4>Sources:</h4>
              <ul>
                {sources.map((source, idx) => (
                  <li key={idx} className={styles.sourceItem}>
                    <strong>{source.title}</strong>: {source.content}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <form onSubmit={handleSubmit} className={styles.chatInputForm}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about Physical AI & Humanoid Robotics..."
              className={styles.chatInput}
              disabled={isLoading}
            />
            <button 
              type="submit" 
              className={styles.sendButton}
              disabled={isLoading || !inputValue.trim()}
            >
              Send
            </button>
          </form>
        </div>
      )}
    </>
  );
};

export default ChatBot;