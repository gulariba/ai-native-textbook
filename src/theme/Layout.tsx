import React from 'react';
import Layout from '@theme-original/Layout';

export default function CustomLayout(props) {
  return (
    <>
      <Layout {...props}>
        {props.children}
      </Layout>
    </>
  );
}