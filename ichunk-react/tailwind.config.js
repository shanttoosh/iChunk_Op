/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#282828',
        primaryDark: '#161616',
        secondary: '#424242',
        secondaryDark: '#4e332a',
        highlight: '#e75f33',
        textPrimary: '#ffffff',
        textSecondary: '#999999',
        textTertiary: '#a3a3a3',
        border: '#ffffff1f',
        success: '#d8fc77',
        danger: '#dc143c',
        warning: '#fbb03b',
      },
      borderRadius: {
        card: '6px',
      },
      transitionDuration: {
        DEFAULT: '200ms',
      },
      transitionTimingFunction: {
        DEFAULT: 'ease',
      },
    },
  },
  plugins: [],
};
