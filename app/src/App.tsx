import {
  createTheme,
  CssBaseline,
  Grid,
  IconButton,
  ThemeProvider,
} from '@mui/material';
import LightModeOutlinedIcon from '@mui/icons-material/LightModeOutlined';
import DarkModeOutlinedIcon from '@mui/icons-material/DarkModeOutlined';
import { useEffect, useState } from 'react';
import IframeResizer from 'iframe-resizer-react';

export default function App() {
  const nameOfSpace = 'itsyoboieltr/anpr-yolov7';
  const [darkmode, setDark] = useState(() => {
    const stored = localStorage.getItem('darkmode');
    if (stored) return stored === 'true';
    else return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  const setDarkmode = (darkmode: boolean) => {
    localStorage.setItem('darkmode', `${darkmode}`);
    setDark(darkmode);
  };
  const theme = createTheme({
    palette: {
      mode: darkmode ? 'dark' : 'light',
      background: { default: darkmode ? '#0c0e19' : '#ffffff' },
    },
  });
  const [opacityDark, setOpacityDark] = useState(darkmode ? 1 : 0);
  const [opacityLight, setOpacityLight] = useState(darkmode ? 0 : 1);

  useEffect(() => {
    setOpacityDark(0);
    setOpacityLight(0);
  }, [darkmode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Grid container justifyContent={'center'}>
        <Grid item xs={12}>
          <IconButton
            size={'large'}
            edge={'end'}
            color={'inherit'}
            onClick={() => {
              if (darkmode) setOpacityDark(0);
              else setOpacityLight(0);
              setTimeout(() => {
                setDarkmode(!darkmode);
              }, 150);
            }}
            sx={{ float: 'right', mr: 1, mb: -6, zIndex: 1 }}>
            {darkmode ? <LightModeOutlinedIcon /> : <DarkModeOutlinedIcon />}
          </IconButton>
          {darkmode ? (
            <IframeResizer
              src={`https://hf.space/embed/${nameOfSpace}/+?__theme=dark`}
              frameBorder={'0'}
              allow={
                'accelerometer; ambient-light-sensor; autoplay; battery; camera; clipboard-write; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking'
              }
              sandbox={
                'allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads'
              }
              style={{
                width: '1px',
                minWidth: '100%',
                opacity: opacityDark,
                transitionDuration: '300ms',
                transitionProperty: 'opacity',
                transitionTimingFunction: 'ease-in-out',
              }}
              onLoad={() => setOpacityDark(1)}
            />
          ) : (
            <IframeResizer
              src={`https://hf.space/embed/${nameOfSpace}/+?__theme=light`}
              frameBorder={'0'}
              allow={
                'accelerometer; ambient-light-sensor; autoplay; battery; camera; clipboard-write; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking'
              }
              sandbox={
                'allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads'
              }
              style={{
                width: '1px',
                minWidth: '100%',
                opacity: opacityLight,
                transitionDuration: '300ms',
                transitionProperty: 'opacity',
                transitionTimingFunction: 'ease-in-out',
              }}
              onLoad={() => setOpacityLight(1)}
            />
          )}
        </Grid>
      </Grid>
    </ThemeProvider>
  );
}
