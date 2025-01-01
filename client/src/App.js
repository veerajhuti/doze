import React, { useState, useEffect } from 'react'

function ChipButton({ isTracking, toggleTracking }) {

  var endingAudio = new Audio("/satisfied.m4a");
  endingAudio.volume = 1;

  return (
    <div
      style={{
        position: 'absolute',
        bottom: '20px',
        right: '20px',
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'flex-start',
        zIndex: 10,
      }}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'flex-end',
          marginRight: '20px',
        }}
      >
        <div
          style={{
            color: 'white',
            font: '700 24px/1 "Roboto", sans-serif',
            fontSize: '24px',
            fontWeight: 'bold',
            letterSpacing: '2px',
            textTransform: 'uppercase',
            marginBottom: '8px',
            textShadow: '0 0 5px rgba(255, 255, 255, 0.8), 0 0 10px rgba(255, 255, 255, 0.6), 0 0 15px rgba(255, 255, 255, 0.4)',
          }}
        >
          {isTracking ? 'Stop Tracking' : 'Start Tracking'}
        </div>

        <div
          style={{
            width: '200px',
            height: '2px',
            background: 'linear-gradient(90deg, rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.2))',
            margin: '0 auto',
            boxShadow: '0 0 8px rgba(255, 255, 255, 1), 0 0 20px rgba(255, 255, 255, 0.8)',
          }}
        />
      </div>

      <button
        onClick={() => {
          toggleTracking();
          if (isTracking) {
            endingAudio.play();
          }
        }}
        style={{
          border: 'none',
          width: '120px',
          height: '120px',
          transform: `rotate(${isTracking ? '-90deg' : '0deg'})`,
          backgroundImage: 'url(/chip2.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          borderRadius: '50%',
          filter: 'drop-shadow(0 0 8px rgba(255, 255, 255, 1))',
          boxShadow: '0 0 15px rgba(255, 255, 255, 0.8)',
        }}
      />
    </div>
  );
}

function App() {

  const [data, setData] = useState({})
  const [isTracking, setIsTracking] = useState(false);

  useEffect(() => {
    fetch('/members')
      .then((res) => {
        if (!res.ok) {
          throw new Error('Network response was not ok');
        }
        return res.text();
      })
      .then((data) => {
        try {
          const jsonData = JSON.parse(data);
          setData(jsonData);
          console.log(jsonData);
        } catch (error) {
          console.error('Failed to parse JSON:', error);
        }
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
      });
  }, []);

  const toggleTracking = () => {
    setIsTracking(!isTracking);
  };

  var greetingAudio = new Audio("/greeting.m4a");

  return (
    <div
      onMouseDown={(e) => {
        if (!e.target.closest('button')) {
          greetingAudio.play();
        }
      }}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'radial-gradient(circle at center, #fff, #fff 50%, #aaa)',
        backgroundSize: 'cover',
        backgroundRepeat: 'no-repeat',
        margin: 0,
        padding: 0,
        transition: 'all 1s ease',
      }}
    >
      <style>
        {`
          @keyframes smile {
            0%, 50% {
              background-position: 0 0;
            }
            85%, 95% {
              background-position: 0 30%;
            }
            100% {
              background-position: 0 0;
            }
          }
        `}
      </style>

      <div
        style={{
          borderBottom: '1.5em solid #000',
          position: 'absolute',
          top: '50%',
          left: '50%',
          width: '50%',
          transform: isTracking ? 'translate(-50%, -900%)' : 'translate(-50%, -40%)',
          transition: 'all 1s ease',
        }}
      >
        <div
          style={{
            animation: 'smile 6s infinite',
            background: 'linear-gradient(to top, #efefef, #efefef 50%, #000 50%, #000)',
            backgroundPosition: '0 0',
            backgroundSize: '200% 200%',
            borderRadius: '50%',
            position: 'absolute',
            width: '12em',
            height: '12em',
            left: '-9em',
            top: '-6em',
            transform: 'skewX(-4deg)',
            transition: 'all 1s ease',
          }}
        />
        <div
          style={{
            animation: 'smile 6s 0.1s infinite',
            background: 'linear-gradient(to top, #efefef, #efefef 50%, #000 50%, #000)',
            backgroundPosition: '0 0',
            backgroundSize: '200% 200%',
            borderRadius: '50%',
            position: 'absolute',
            width: '12em',
            height: '12em',
            right: '-9em',
            top: '-6em',
            transform: 'skewX(4deg)',
          }}
        />

        {isTracking && (
          <div
            style={{
              position: 'absolute',
              bottom: '20px',
              left: '50%',
              transform: 'translateX(-50%)',
              transition: 'all 1s ease',
            }}
          >
            <video
              autoPlay
              muted
              width="320"
              height="240"
              style={{
                borderRadius: '10px',
                boxShadow: '0 0 10px rgba(255, 255, 255, 0.8)',
              }}
            >
              <source src="/webcam-feed.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        )}
      </div>
      <ChipButton isTracking={isTracking} toggleTracking={toggleTracking} />
    </div>
  );
}

export default App