import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './ChapterCard.module.css';

type ChapterCardProps = {
  title: string;
  description: string;
  index: number;
};

const ChapterCard: React.FC<ChapterCardProps> = ({ title, description, index }) => {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      if (!isHovered) return;
      
      const card = document.querySelectorAll(`.${styles.chapterCard}`)[index] as HTMLElement;
      if (!card) return;
      
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      
      const rotateY = (x - centerX) / 25;
      const rotateX = (centerY - y) / 25;
      
      const boundedX = Math.min(Math.max(rotateX, -5), 5);
      const boundedY = Math.min(Math.max(rotateY, -5), 5);
      
      setRotation({ x: boundedX, y: boundedY });
      
      // Add floating effect
      const moveX = (x / rect.width) * 10 - 5;
      const moveY = (y / rect.height) * 10 - 5;
      
      setPosition({ x: moveX, y: moveY });
    };

    const handleEnter = () => {
      setIsHovered(true);
    };

    const handleLeave = () => {
      setIsHovered(false);
      setRotation({ x: 0, y: 0 });
      setPosition({ x: 0, y: 0 });
    };

    const card = document.querySelectorAll(`.${styles.chapterCard}`)[index];
    if (card) {
      card.addEventListener('mousemove', handleMove);
      card.addEventListener('mouseenter', handleEnter);
      card.addEventListener('mouseleave', handleLeave);
    }

    return () => {
      if (card) {
        card.removeEventListener('mousemove', handleMove);
        card.removeEventListener('mouseenter', handleEnter);
        card.removeEventListener('mouseleave', handleLeave);
      }
    };
  }, [isHovered, index]);

  return (
    <div 
      className={clsx(styles.chapterCard)}
      style={{
        transform: `perspective(1000px) rotateX(${rotation.x}deg) rotateY(${rotation.y}deg) translate3d(${position.x}px, ${position.y}px, 20px)`,
        transition: isHovered ? 'none' : 'transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease'
      }}
    >
      <h3 className={styles.cardTitle}>{title}</h3>
      <p className={styles.cardDescription}>{description}</p>
    </div>
  );
};

export default ChapterCard;