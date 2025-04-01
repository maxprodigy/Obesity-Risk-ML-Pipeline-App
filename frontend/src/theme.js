import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#FFD700',
    },
    secondary: {
      main: '#B8860B',
    },
    background: {
      default: '#000000',
      paper: '#000000',
    },
    text: {
      primary: '#FFD700',
      secondary: 'rgba(255, 215, 0, 0.7)',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 700,
      color: '#FFD700',
    },
    h6: {
      fontWeight: 600,
      color: '#FFD700',
    },
    body1: {
      color: '#FFD700',
    },
    body2: {
      color: 'rgba(255, 215, 0, 0.7)',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
        },
        contained: {
          backgroundColor: '#000000',
          color: '#FFD700',
          border: '2px solid #FFD700',
          '&:hover': {
            backgroundColor: '#FFD700',
            color: '#000000',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: '#000000',
          border: '1px solid #FFD700',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#000000',
          borderBottom: '1px solid #FFD700',
        },
      },
    },
  },
});

export default theme; 