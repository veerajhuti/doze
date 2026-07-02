import React, { useState, useEffect, useRef } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import Button from '@mui/joy/Button';
import "react-toastify/dist/ReactToastify.css";

function Vanilla({ onStartSim }) {
  const [isTracking, setIsTracking] = useState(false);
  const apiUrl = 'https://docker-server-xdlv.onrender.com';
  // for testing
  // const apiUrl = 'http://localhost:4000';
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const lastAlertRef = useRef(0); // timestamp of last drowsiness alert
  
  const alarm = React.useMemo(() =>
    new Audio(process.env.PUBLIC_URL + "/alarm-clock-short-6402.mp3"), 
  [] );
  
  useEffect(() => {
  if (isTracking) {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        console.error("Error accessing webcam:", err);
        toast.error("Could not access webcam.");
        setIsTracking(false);
      });
  } else {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  }
  }, [isTracking]);
  
  useEffect(() => {
  let intervalId;
  
  if (isTracking) {
    intervalId = setInterval(() => {
      if (!videoRef.current || !canvasRef.current) return;
  
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
  
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageDataUrl = canvas.toDataURL("image/jpeg");
  
      fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageDataUrl }),
      })
        .then((res) => res.json())
        .then((data) => {
          console.log("Drowsiness prediction:", data);
          const now = Date.now();
          if (data.is_drowsy && now - lastAlertRef.current > 5000) {
            lastAlertRef.current = now;
            alarm.play().catch(() => {
              // handle autoplay block on some browsers
            });
            toast.error("You seem drowsy. Please take a break.", {
              autoClose: 4000,
            });
          }
        })
        .catch((err) => {
          console.error("Prediction error:", err);
        });
    }, 1000);
  }
  
  return () => clearInterval(intervalId);
  }, [isTracking, apiUrl, alarm]);
  
  const stopWebcam = () => {
    if (videoRef.current) {
      const stream = videoRef.current.srcObject;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      videoRef.current.srcObject = null;
      videoRef.current.pause();
      videoRef.current.load();
    }
  };
  
  const toggleTracking = () => {
    if (isTracking) {
      stopWebcam();
    }
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
          width: '350px',
          height: 'auto',
        }}
      >
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{ width: '100%', height: 'auto', borderRadius: '10px' }}
        />
        <canvas
          ref={canvasRef}
          width={450}
          height={300}
          style={{ display: 'none' }}
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
                background: 'linear-gradient(to bottom, #a3a6b6, #8873FF)',
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
  
          <div style={{ marginTop: '15px' }}>
            <Button
              variant="plain"
              sx={{
                background: 'linear-gradient(to bottom, #ffc884, #8873FF)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                padding: 0,
                minWidth: 'auto',
              }}
              onClick={onStartSim}
            >
              Try the Driving Simulator
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

export default Vanilla;