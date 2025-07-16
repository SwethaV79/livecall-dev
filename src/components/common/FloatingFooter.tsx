import React from 'react';

const FloatingFooter: React.FC = () => (
    <footer
        className="footer fixed bottom-0 left-0 w-full flex-shrink-0 px-0"
        style={{
            backgroundColor: 'rgb(16 24 39)',
            backdropFilter: 'blur(8px)',
            fontSize: '0.75rem',
            lineHeight: 1,
            height: 20,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'hsl(0, 0%, 90.9%)',
            zIndex: 50,
        }}
    >
        <p style={{ margin: 0 }}>© 2025 Wipro. All rights reserved.</p>
    </footer>
);

export default FloatingFooter;
