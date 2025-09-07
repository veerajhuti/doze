import React, { useState, useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import Button from '@mui/joy/Button';

function App() {

  const [isTracking, setIsTracking] = useState(false);
  const apiUrl = 'http://backend:4000';
  
  const alarm_sound = React.useMemo(() =>
    new Audio(process.env.PUBLIC_URL + "/alarm-clock-short-6402.mp3"), 
  [] );

  useEffect(() => {
    if (isTracking) {
      fetch(`${apiUrl}/webcam`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }).catch(error => {
        console.error('Error fetching webcam data:', error);
      });
    } else {
      fetch(`${apiUrl}/stop_webcam`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }).catch(error => {
        console.error('Error stopping webcam:', error);
      });
    }
  }, [isTracking, apiUrl]);

  useEffect(() => {
    if (isTracking) {
      fetch(`${apiUrl}/check_drowsiness`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then(data => {
          console.log('Drowsiness data:', data.is_drowsy);
          if (data.is_drowsy) {
            alarm_sound.play();
            toast.error('You seem drowsy. Please take a break.', 'Drowsiness Alert', 4000);
          }
        })
        .catch(error => {
          console.error('Error fetching webcam data:', error);
        });
      }
}, [isTracking, apiUrl, alarm_sound]);  

  const toggleTracking = () => {
    setIsTracking(!isTracking);
  };

  return (
    <div style={{ position: 'relative', height: '100vh', overflow: 'hidden' }}>
    
    <ToastContainer />
   
    <video
      autoPlay
      muted
      loop
      playsInline
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '120vh',
        zIndex: -1,
        objectFit: 'fill',
        filter: 'blur(3px)',
      }}
    >
    <source src={process.env.PUBLIC_URL + '/gif.mp4'} type="video/mp4" />
    Your browser does not support the video tag.
  </video>

    {isTracking && (
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          boxShadow: '0 0 20px rgba(169, 169, 169, 0.5)',
          filter: 'drop-shadow(0 0 10px rgba(169, 169, 169, 0.8))',
          borderRadius: '10px',
          zIndex: 20,
          width: '450px',
          height: 'auto',
        }}
      >
        <img
          src={`${apiUrl}/webcam`}
          alt="Webcam Feed"
          style={{ width: '450px', height: 'auto', borderRadius: '10px' }}
        />
      </div>
    )}
    <div
      style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        background: 'rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(15px)',
        WebkitBackdropFilter: 'blur(15px)',
        padding: '40px',
        borderRadius: '20px',
        width: '90%',
        maxWidth: '520px',
        color: 'white',
        fontFamily: '"Baloo", sans-serif',
        textAlign: 'center',
        zIndex: 0,
        border: '1px solid rgba(255, 255, 255, 0.2)',
        boxShadow: '0 4px 30px rgba(0, 0, 0, 0.2)',
        transition: 'all 0.5s ease',
      }}
    >
      <h1
        style={{
          fontSize: '32px',
          marginBottom: '20px',
          background: 'linear-gradient(to bottom, #8496FF, #8873FF)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
        >
        Doze
      </h1>
      
      <div
        style={{
          opacity: isTracking ? 0 : 1,
          transition: 'opacity 0.5s ease',
          pointerEvents: isTracking ? 'none' : 'auto',
        }}
        >
          <p>
            Doze watches for signs of drowsiness using your webcam. You'll be
            alerted if you appear sleepy.
          </p>
          <p>Use this while studying, working, or driving long hours.</p>
          <p>Please allow camera access. No data is stored or uploaded.</p>

          <Button
            onClick={toggleTracking}
            variant="solid"
            sx={{
              fontSize: '18px',
              marginTop: '20px',
              padding: '10px 20px',
              backgroundColor: 'white',
              // fontWeight: 'bold',
              '&:hover': {
                backgroundColor: '#e6e8ff', // light purpleish background on hover
              },
              cursor: 'pointer',
            }}
          >
            <span
              style={{
                background: 'linear-gradient(to bottom, #8496FF, #8873FF)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                display: 'inline-block',
                width: '100%',
              }}
            >
              Start Tracking
            </span>
          </Button>

          <div style={{ marginTop: '15px' }}>
            <Button
              variant="plain"
              sx={{
                background: 'linear-gradient(to bottom, #8496FF, #8873FF)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                padding: 0,
                minWidth: 'auto',
              }}
              onClick={() =>
                window.open(
                  'mailto:veerajhuti@gmail.com?subject=Feedback for Doze',
                  '_blank'
                )
              }
            >
              Feedback / Questions
            </Button>
          </div>
        </div>

        {isTracking && (
          <Button
            onClick={toggleTracking}
            variant="solid"
            color="danger"
            sx={{
              fontSize: '18px',
              marginTop: '20px',
              padding: '10px 20px',
            }}
          >
            Stop Tracking
          </Button>
        )}
      </div>
    </div>
  );
}

export default App;