import React from 'react';
import Layout from '@theme-original/Layout';
import ChatWidget from '../components/ChatWidget/ChatWidget';

export default function CustomLayout(props) {
  return (
    <>
      <Layout {...props}>
        {props.children}
        <ChatWidget />
      </Layout>
    </>
  );
}